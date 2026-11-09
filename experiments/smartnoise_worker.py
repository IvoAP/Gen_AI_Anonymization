import argparse
from importlib.metadata import version
import json
from pathlib import Path
import random
import traceback
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from gen_ai_anonymization.baseline import AIMBaseline, DPCTGANBaseline, MSTBaseline, MWEMBaseline
from gen_ai_anonymization.dptal.classification import fit_and_evaluate
from gen_ai_anonymization.dptal.membership import MembershipConfig, MembershipData
from .smartnoise import quantize_features


def create_baseline(method: str, config: dict):
    epsilon = config["epsilon"]
    if method == "mwem":
        return MWEMBaseline(epsilon, iterations=config["mwem_iterations"],
                            split_factor=config["mwem_split_factor"], verbose=config["verbose"] >= 2)
    if method == "mst":
        return MSTBaseline(epsilon, delta=config["delta"], verbose=config["verbose"] >= 2)
    if method == "aim":
        return AIMBaseline(epsilon, delta=config["delta"], rounds=config["aim_rounds"],
                           verbose=config["verbose"] >= 2)
    if method == "dpctgan":
        return DPCTGANBaseline(epsilon, delta=config["delta"], epochs=config["dpctgan_epochs"],
                               sigma=config["dpctgan_sigma"], batch_size=config["batch_size"],
                               cuda=False, verbose=config["verbose"] >= 2)
    raise ValueError("Unsupported SmartNoise method.")


def log_progress(config: dict, method: str, message: str) -> None:
    if config["verbose"]:
        print(f"[SmartNoise:{method}] {message}", flush=True)


def public_transformer(columns: list[str], bins: int, method: str):
    from snsynth.transform import MinMaxTransformer, TableTransformer

    features = [column for column in columns if column != "label"]
    continuous = method == "dpctgan"
    public = pd.DataFrame({
        column: np.linspace(-1, 1, bins) if continuous else np.arange(bins) for column in features
    })
    public["label"] = np.arange(bins) % 2
    public = public[columns]
    transformer = TableTransformer.create(
        public, style="gan" if continuous else "cube",
        categorical_columns=["label"] if continuous else columns,
        continuous_columns=features if continuous else [], nullable=False,
        constraints={column: MinMaxTransformer(lower=-1, upper=1) for column in features} if continuous else None,
    )
    transformer.fit(public)
    return transformer


def run_worker(inputs: Path, config: dict, output: Path) -> dict:
    torch.set_num_threads(config["threads"])
    with np.load(inputs, allow_pickle=False) as data:
        train_features = data["train_features"]
        train_labels = data["train_labels"]
        test_features = data["test_features"]
        test_labels = data["test_labels"]
        nonmember_features = data["nonmember_features"] if not config.get("skip_membership", True) else None
        nonmember_labels = data["nonmember_labels"] if nonmember_features is not None else None
    if (not np.isfinite(train_features).all() or not np.isfinite(test_features).all()
            or np.max(np.abs(train_features)) > 1 or np.max(np.abs(test_features)) > 1):
        raise ValueError("Worker inputs must be original preprocessed features bounded to [-1,1], not latents.")
    feature_columns = [f"feature_{index}" for index in range(train_features.shape[1])]
    report = {"backends": [], "evaluations": [], "membership_inference": [], "versions": {
        name: version(name) for name in ("smartnoise-synth", "opacus", "torch", "scikit-learn")
    }}
    for index, method in enumerate(config["methods"], start=1):
        method_start = perf_counter()
        log_progress(config, method, f"START {index}/{len(config['methods'])} | Seed: {config['seed']} | "
                     f"Training records: {len(train_features)} | Original features: {len(feature_columns)} | Bins: {config['bins']}")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        backend_report = {"method": method, "status": "running"}
        try:
            continuous = method == "dpctgan"
            method_train = train_features if continuous else quantize_features(train_features, config["bins"])
            method_test = test_features if continuous else quantize_features(test_features, config["bins"])
            training = pd.DataFrame(method_train, columns=feature_columns)
            training["label"] = train_labels
            if method == "mwem":
                if config["mwem_split_factor"] < 2:
                    raise ValueError("MWEM partition width must be at least two to retain a label-feature dependency.")
                training = training[["label", *feature_columns]]
            columns = list(training.columns)
            baseline = create_baseline(method, config)
            log_progress(config, method, "Preparing public-domain transformer")
            transformer = public_transformer(columns, config["bins"], method)
            cardinality = list(transformer.cardinality)
            log_progress(config, method, f"Training started | Parameters: {baseline.parameters}")
            baseline.fit(training, transformer=transformer)
            log_progress(config, method, f"Training completed | Time: {baseline.training_seconds:.2f}s")
            if list(transformer.cardinality) != cardinality:
                raise RuntimeError("The backend changed the public domain during fit.")
            log_progress(config, method, f"Sampling started | Requested records: {config['synthetic_samples']}")
            synthetic = baseline.sample(config["synthetic_samples"])
            log_progress(config, method, f"Sampling completed | Time: {baseline.sampling_seconds:.2f}s")
            targets = synthetic["label"].to_numpy(dtype=np.int64)
            features = synthetic[feature_columns].to_numpy(dtype=np.float32)
            if not np.isfinite(features).all() or not np.isin(targets, [0, 1]).all():
                raise RuntimeError("The backend generated invalid values or labels.")
            if continuous and (np.abs(features) > 1.000001).any():
                raise RuntimeError("The backend generated values outside the public continuous bounds.")
            if not continuous and ((features < 0).any() or (features >= config["bins"]).any()):
                raise RuntimeError("The backend generated values outside the public domain.")
            backend_report.update(baseline.get_report())
            train_representation = "synthetic_features_continuous" if continuous else "synthetic_features_public_bins"
            test_representation = "real_features_continuous" if continuous else "real_features_public_bins"
            scenario = "Trained on independent SmartNoise synthetic features; tested on the same real test records"
            membership = None
            if nonmember_features is not None:
                method_nonmembers = nonmember_features if continuous else quantize_features(nonmember_features, config["bins"])
                membership = MembershipData(method_train, train_labels, method_nonmembers, nonmember_labels,
                                            "synthesizer_training_records")
            if len(np.unique(targets)) != 2:
                backend_report.update(status="skipped", reason="Synthetic labels contain only one class.")
                log_progress(config, method, f"SKIPPED | {backend_report['reason']}")
            else:
                for classifier in config["classifiers"]:
                    log_progress(config, method, f"Classifier {classifier} started | Test records: {len(test_labels)}")
                    result = fit_and_evaluate(
                        classifier, features, targets, method_test, test_labels,
                        seed=config["seed"], max_iter=config["classifier_max_iter"], verbose=config["verbose"],
                        membership_data=membership,
                        membership_config=MembershipConfig(max_records_per_group=config.get("mia_max_records", 2000),
                                                           target_fpr=config.get("mia_target_fpr", 0.01), seed=config["seed"]),
                    )
                    for attack in result.pop("membership_inference", []):
                        report["membership_inference"].append({
                            "method": method, "classifier": classifier, "evaluation_scenario": scenario, **attack,
                        })
                        log_progress(config, method, f"MIA {attack['attack']} | Status: {attack['status']} | "
                                     f"AUC: {attack.get('roc_auc', 'unavailable')}")
                    report["evaluations"].append({
                        "method": method, "classifier": classifier,
                        "train_representation": train_representation, "test_representation": test_representation,
                        "evaluation_scenario": scenario, "feature_bins": None if continuous else config["bins"],
                        "train_records": len(targets), "test_records": len(test_labels),
                        "input_dimensions": features.shape[1],
                        "baseline_training_seconds": baseline.training_seconds,
                        "baseline_sampling_seconds": baseline.sampling_seconds, **result,
                    })
                    auc = "unavailable" if result["roc_auc"] is None else f"{result['roc_auc']:.4f}"
                    log_progress(config, method, f"Classifier {classifier} completed | "
                                 f"Accuracy: {result['accuracy']:.4f} | F1: {result['f1']:.4f} | "
                                 f"Macro-F1: {result['macro_f1']:.4f} | Precision: {result['precision_positive']:.4f} | "
                                 f"Recall: {result['recall_positive']:.4f} | FPR: {result['false_positive_rate']:.4f} | "
                                 f"ROC-AUC: {auc} | Cohen kappa: {result['cohen_kappa']:.4f} | "
                                 f"Training: {result['classifier_training_seconds']:.2f}s | "
                                 f"Inference: {result['classifier_inference_seconds']:.2f}s | "
                                 f"Converged: {result['converged']}")
                backend_report["status"] = "completed"
        except Exception as error:
            backend_report.update(status="failed", error=f"{type(error).__name__}: {error}",
                                  traceback=traceback.format_exc())
            log_progress(config, method, f"FAILED | {backend_report['error']}")
        report["backends"].append(backend_report)
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        log_progress(config, method, f"END | Status: {backend_report['status']} | "
                     f"Total time: {perf_counter() - method_start:.2f}s | Report saved: {output.name}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal worker for the isolated SmartNoise environment.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_worker(args.input, json.loads(args.config.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()