import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import perf_counter
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch

from gen_ai_anonymization.dptal.classification import CLASSIFIERS, fit_and_evaluate
from gen_ai_anonymization.dptal.data import load_dataset
from gen_ai_anonymization.dptal.membership import MembershipConfig, MembershipData


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "ddos_csv_100000.csv"
EVALUATION_SPACES = {
    "real_features": ("real_features", "real_features", "Trained on real features; tested on real X_test"),
    "real_latents": ("real_latents", "real_latents", "Trained on real latents; tested on X_test_latent"),
    "synthetic_latents": ("synthetic_latents", "real_latents", "Trained on synthetic latents; tested on X_test_latent"),
}
METRICS = (
    "accuracy", "f1", "macro_f1", "precision_positive", "cohen_kappa",
    "recall_positive", "false_positive_rate", "roc_auc",
    "classifier_training_seconds", "classifier_inference_seconds",
)


def parse_args(
    argv: list[str] | None = None, *, mode: str = "combined",
    configure_parser: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare classifiers trained on real features, encoded records and synthetic latents."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--label", default="Label")
    parser.add_argument("--positive-label", type=int, choices=(0, 1), default=1,
                        help="Original label treated as positive for F1 and recall; verify the attack mapping.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--noise-multipliers", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--prior-separation", type=float, default=1.0)
    parser.add_argument("--synthetic-samples", type=int, default=10000,
                        help="Fixed sample count, independent of private class frequencies.")
    parser.add_argument("--classifiers", choices=CLASSIFIERS, nargs="+", default=list(CLASSIFIERS))
    parser.add_argument("--classifier-max-iter", type=int, default=200)
    parser.add_argument("--skip-membership", action="store_true", help="Disable membership inference audits.")
    parser.add_argument("--mia-max-records", type=int, default=2000,
                        help="Maximum original records per membership group, split between calibration and evaluation.")
    parser.add_argument("--mia-target-fpr", type=float, default=0.01,
                        help="Threshold calibration FPR; observed evaluation FPR is reported separately.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; saved for timing reproducibility.")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--output", type=Path, default=Path("results") / mode)
    parser.add_argument("--save-model", action="store_true", help="Save encoder weights for internal research use.")
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args(argv)
    args.experiment = mode
    if mode in ("dptal", "smartnoise"):
        args.skip_membership = True
    elif mode == "mia":
        if args.skip_membership:
            parser.error("The MIA experiment cannot disable membership inference.")
        args.skip_membership = False
    for name in ("epochs", "batch_size", "latent_dim", "hidden_dim", "anchors", "threads", "classifier_max_iter"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    for name in ("learning_rate", "max_grad_norm", "prior_separation"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if not 0 < args.delta < 1:
        parser.error("--delta must be between zero and one.")
    for name in ("kl_weight", "geometry_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative.")
    if any(not math.isfinite(noise) or noise < 0 for noise in args.noise_multipliers):
        parser.error("Noise multipliers must be finite and nonnegative.")
    if args.synthetic_samples < 2:
        parser.error("--synthetic-samples must be at least two.")
    if args.max_rows is not None and args.max_rows < 100:
        parser.error("--max-rows must be at least 100.")
    if args.mia_max_records < 8:
        parser.error("--mia-max-records must be at least eight.")
    if not 0 < args.mia_target_fpr < 1:
        parser.error("--mia-target-fpr must be between zero and one.")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32).")
    args.noise_multipliers = list(dict.fromkeys(args.noise_multipliers))
    args.seeds = list(dict.fromkeys(args.seeds))
    args.classifiers = list(dict.fromkeys(args.classifiers))
    return args


def save_results(output: Path, rows: list[dict], report: dict) -> None:
    output.joinpath("report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    attacks = report.get("membership_inference", [])
    if attacks:
        attack_frame = pd.DataFrame(attacks)
        attack_frame.to_csv(output / "membership_inference.csv", index=False)
        completed = attack_frame[attack_frame["status"] == "completed"]
        if not completed.empty:
            grouped_attacks = completed.groupby([
                "representation", "classifier", "noise_multiplier", "membership_scope", "attack"
            ], dropna=False)
            summary_attacks = grouped_attacks[["roc_auc", "accuracy", "f1", "tpr", "fpr", "advantage"]].agg(["mean", "std"])
            summary_attacks.columns = [f"{metric}_{statistic}" for metric, statistic in summary_attacks.columns]
            summary_attacks["seed_count"] = grouped_attacks.size()
            summary_attacks.to_csv(output / "membership_summary.csv")
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    grouped = frame.groupby([
        "representation", "train_representation", "test_representation", "evaluation_scenario",
        "classifier", "noise_multiplier",
    ], dropna=False)
    summary = grouped[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary["seed_count"] = grouped.size()
    summary.to_csv(output / "summary.csv")


def evaluate_representation(
    representation: str, train_features: np.ndarray, train_labels: np.ndarray,
    test_features: np.ndarray, test_labels: np.ndarray, args: argparse.Namespace,
    context: dict, rows: list[dict], report: dict, output: Path,
    membership_data: MembershipData | None = None,
    spaces: tuple[str, str, str] | None = None,
) -> None:
    train_representation, test_representation, scenario = spaces or EVALUATION_SPACES[representation]
    for name in args.classifiers:
        if args.verbose:
            print(f"[Classifier] {name} | {scenario} | "
                  f"Seed: {context['seed']} | Noise: {context['noise_multiplier']}", flush=True)
        result = fit_and_evaluate(
            name, train_features, train_labels, test_features, test_labels,
            seed=context["seed"], max_iter=args.classifier_max_iter, verbose=args.verbose,
            membership_data=membership_data,
            membership_config=MembershipConfig(
                max_records_per_group=args.mia_max_records, target_fpr=args.mia_target_fpr,
                seed=context["seed"],
            ),
        )
        parameters = result.pop("classifier_params")
        messages = result.pop("warnings")
        attacks = result.pop("membership_inference", [])
        row = {
            **context, "representation": representation, "classifier": name,
            "train_representation": train_representation, "test_representation": test_representation,
            "evaluation_scenario": scenario,
            "train_records": len(train_labels), "test_records": len(test_labels),
            "input_dimensions": train_features.shape[1], "positive_original_label": args.positive_label,
            **result,
        }
        rows.append(row)
        report["classifiers"].append({**row, "parameters": parameters, "warnings": messages})
        for attack in attacks:
            report["membership_inference"].append({
                **context, "representation": representation, "classifier": name,
                "evaluation_scenario": scenario, "positive_original_label": args.positive_label,
                **attack,
            })
            if args.verbose:
                if attack["status"] == "completed":
                    print(f"[Membership] {attack['attack']} | Scope: {attack['membership_scope']} | "
                          f"AUC: {attack['roc_auc']:.4f} | TPR: {attack['tpr']:.4f} | "
                          f"Observed FPR: {attack['fpr']:.4f}", flush=True)
                    if attack["low_fpr_sample_warning"]:
                        print("[Membership warning] Too few nonmembers for a stable low-FPR estimate.", flush=True)
                else:
                    print(f"[Membership] Skipped: {attack['reason']}", flush=True)
        save_results(output, rows, report)
        if args.verbose:
            print(f"[Evaluation] Accuracy: {result['accuracy']:.4f} | F1: {result['f1']:.4f} | "
                f"Macro-F1: {result['macro_f1']:.4f} | Precision: {result['precision_positive']:.4f} | "
                f"Recall: {result['recall_positive']:.4f} | FPR: {result['false_positive_rate']:.4f} | "
                f"ROC-AUC: {result['roc_auc']:.4f} | "
                  f"Cohen kappa: {result['cohen_kappa']:.4f} | "
                  f"Training: {result['classifier_training_seconds']:.2f}s", flush=True)


def run_experiment(args: argparse.Namespace, *, baseline_comparison: Callable | None = None) -> Path:
    from gen_ai_anonymization.dptal import ModelConfig, TrainingConfig, train_generator
    from gen_ai_anonymization.dptal.training import encode_features, generate_latents

    torch.set_num_threads(args.threads)
    start = perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output / run_id
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "started_at_utc": run_id,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "torch": str(torch.__version__)},
        "protocol": {
            "split": "Seeded random record partition: 60% train, 20% attack nonmembers, 20% classification test.",
            "preprocessing": "Fixed signed log1p / 20, clipped to [-1,1]; missing/infinite values mapped to zero.",
            "test_encoding": "Deterministic encoder posterior mean; labels are never passed to the encoder.",
            "evaluation_spaces": "Classifiers are trained and tested in matching feature spaces; latent tests encode held-out real records.",
            "synthetic_sampling": "Balanced labels and fixed conditional Gaussian priors; same seed gives identical samples across noise levels.",
            "privacy": "Record-level generator training only. Baselines, metrics, losses and real encodings are not DP-protected releases.",
            "composition": "Epsilon is reported per generator, not for the combined release of multiple runs.",
            "membership": "Known-label probability attacks; class-matched original members/nonmembers with disjoint calibration/evaluation. No classification-test records are used.",
            "membership_scope": "Synthetic-trained classifiers are audited for original generator membership via the encoder/classifier pipeline, not synthetic sample membership.",
            "limitations": "No temporal/device grouping or duplicate removal; no shadow models, white-box attacks or neighborhood evaluation. Attack failure is not proof of privacy.",
        },
        "splits": [], "generators": [], "classifiers": [], "membership_inference": [],
    }
    rows = []
    save_results(output, rows, report)
    try:
        for seed in args.seeds:
            if args.verbose:
                print(f"[Data] Loading and splitting dataset | Seed: {seed}", flush=True)
            data = load_dataset(args.dataset, args.label, args.max_rows, seed)
            labels = (data.labels == args.positive_label).astype(np.int64)
            train_features, train_labels = data.features[data.train], labels[data.train]
            test_features, test_labels = data.features[data.test], labels[data.test]
            if args.delta >= 1 / len(train_labels):
                raise ValueError("delta must be smaller than 1 / training records.")
            report["splits"].append({
                "seed": seed, "feature_names": data.feature_names,
                "train_records": len(data.train), "reserved_records": len(data.attack_holdout),
                "test_records": len(data.test),
                "train_positive_records": int(train_labels.sum()), "test_positive_records": int(test_labels.sum()),
            })
            evaluate_representation(
                "real_features", train_features, train_labels, test_features, test_labels, args,
                {"seed": seed, "noise_multiplier": None, "epsilon": None, "delta": None,
                 "generator_training_seconds": None, "generation_seconds": None, "encoding_seconds": None},
                rows, report, output,
                membership_data=None if args.skip_membership else MembershipData(
                    train_features, train_labels, data.features[data.attack_holdout], labels[data.attack_holdout],
                    "classifier_training_records",
                ),
            )
            for noise in args.noise_multipliers:
                if args.verbose:
                    print(f"[Generator] Seed: {seed} | Noise multiplier: {noise:g}", flush=True)
                model_config = ModelConfig(
                    latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, anchors=args.anchors,
                    kl_weight=args.kl_weight, geometry_weight=args.geometry_weight,
                    prior_separation=args.prior_separation, seed=seed,
                )
                training_config = TrainingConfig(
                    epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                    noise_multiplier=noise, max_grad_norm=args.max_grad_norm, delta=args.delta,
                    device=args.device, seed=seed, verbose=args.verbose,
                )
                model, generator_report = train_generator(train_features, train_labels, model_config, training_config)
                encoding_start = perf_counter()
                train_latents = encode_features(model, train_features)
                test_latents = encode_features(model, test_features)
                encoding_seconds = perf_counter() - encoding_start
                attack_nonmember_latents = None if args.skip_membership else encode_features(
                    model, data.features[data.attack_holdout]
                )
                synthetic_latents, synthetic_labels, generation_seconds = generate_latents(
                    model, args.synthetic_samples, seed
                )
                report["generators"].append({
                    "seed": seed, "noise_multiplier": noise, **generator_report,
                    "generation_seconds": generation_seconds, "encoding_seconds": encoding_seconds,
                })
                if args.save_model:
                    torch.save(
                        {"state_dict": model.state_dict(), "model_config": generator_report["model_config"],
                         "feature_names": data.feature_names, "positive_original_label": args.positive_label},
                        output / f"generator_seed_{seed}_noise_{noise:g}.pt",
                    )
                context = {
                    "seed": seed, "noise_multiplier": noise,
                    "epsilon": generator_report["epsilon"], "delta": generator_report["delta"],
                    "generator_training_seconds": generator_report["training_seconds"],
                    "generation_seconds": generation_seconds, "encoding_seconds": encoding_seconds,
                }
                for representation, features, targets in (
                    ("real_latents", train_latents, train_labels),
                    ("synthetic_latents", synthetic_latents, synthetic_labels),
                ):
                    evaluate_representation(
                        representation, features, targets, test_latents, test_labels,
                        args, context, rows, report, output,
                        membership_data=None if args.skip_membership else MembershipData(
                            train_latents, train_labels, attack_nonmember_latents, labels[data.attack_holdout],
                            "generator_and_classifier_training_records" if representation == "real_latents"
                            else "generator_training_records",
                        ),
                    )
            if baseline_comparison is not None:
                baseline_comparison(
                    train_features, train_labels, test_features, test_labels,
                    args, {"seed": seed, "feature_names": data.feature_names,
                           "membership_data": None if args.skip_membership else MembershipData(
                               train_features, train_labels, data.features[data.attack_holdout], labels[data.attack_holdout],
                               "synthesizer_training_records",
                           )}, rows, report, output,
                )
        report["status"] = (
            "completed_with_failures" if any(item["status"] == "failed" for item in report.get("baseline_runs", []))
            else "completed"
        )
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["experiment_seconds"] = perf_counter() - start
        save_results(output, rows, report)
    print("\nTEST RESULTS")
    print(pd.DataFrame(rows)[[
        "seed", "noise_multiplier", "representation", "evaluation_scenario", "classifier", "accuracy", "f1", "macro_f1",
        "precision_positive", "recall_positive", "false_positive_rate", "roc_auc", "cohen_kappa", "converged"
    ]].to_string(index=False))
    if report["membership_inference"]:
        print("\nMEMBERSHIP INFERENCE RESULTS")
        completed_attacks = [attack for attack in report["membership_inference"] if attack["status"] == "completed"]
        if completed_attacks:
            print(pd.DataFrame(completed_attacks)[[
                "seed", "noise_multiplier", "representation", "classifier", "membership_scope", "attack",
                "roc_auc", "accuracy", "tpr", "fpr", "advantage",
            ]].to_string(index=False))
        else:
            print("No attacks completed; see report.json for skipped configurations.")
    print(f"\n[Complete] Results saved to: {output.resolve()}")
    print("[Warning] Results are internal research diagnostics, not anonymized or DP-protected releases.")
    return output


if __name__ == "__main__":
    run_experiment(parse_args())