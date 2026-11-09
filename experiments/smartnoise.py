import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Timer

import numpy as np

from .common import parse_args, run_experiment, save_results


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("mwem", "mst", "aim", "dpctgan")
DEFAULT_METHODS = (
    "mwem",
    "mst",
    "aim",
    "dpctgan",
)


def quantize_features(features: np.ndarray, bins: int, bound: float = 1.0) -> np.ndarray:
    if bins < 2 or not math.isfinite(bound) or bound <= 0:
        raise ValueError("Use at least two bins and a finite positive feature bound.")
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite.")
    codes = np.floor((np.clip(features, -bound, bound) + bound) * bins / (2 * bound))
    return np.clip(codes, 0, bins - 1).astype(np.int64)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    environment = Path.home() / ".venvs" / "gen-ai-baselines"
    interpreter = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    parser.add_argument("--baseline-python", type=Path, default=interpreter,
                        help="Interpreter of the separate SmartNoise environment.")
    parser.add_argument("--smartnoise-methods", choices=METHODS, nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--baseline-epsilon", type=float, default=3.0,
                        help="Requested independent SmartNoise synthesis budget, not an encoder-plus-synthesis budget.")
    parser.add_argument("--feature-bins", type=int, default=64,
                        help="Public fixed bins for histogram baselines in the original feature space, bounded to [-1,1].")
    parser.add_argument("--compare-smartnoise", action="store_true",
                        help="Also audit independent SmartNoise pipelines when using experiments.mia.")
    parser.add_argument("--mwem-iterations", type=int, default=10)
    parser.add_argument("--mwem-split-factor", type=int, default=2)
    parser.add_argument("--aim-rounds", type=int, default=None)
    parser.add_argument("--dpctgan-epochs", type=int, default=300)
    parser.add_argument("--dpctgan-sigma", type=float, default=5.0)
    parser.add_argument("--baseline-timeout", type=float, default=None,
                        help="Optional maximum worker runtime in seconds per dataset seed.")


def comparison_args(argv: list[str] | None = None, *, mode: str = "smartnoise") -> argparse.Namespace:
    args = parse_args(argv, mode=mode, configure_parser=configure_parser)
    for name in ("baseline_epsilon", "dpctgan_sigma"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if args.feature_bins < 2:
        raise ValueError("feature_bins must be at least two.")
    for name in ("mwem_iterations", "mwem_split_factor", "dpctgan_epochs"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if "mwem" in args.smartnoise_methods and args.mwem_split_factor < 2:
        raise ValueError("mwem_split_factor must be at least two so the label shares a partition with a feature.")
    if args.baseline_timeout is not None and (
        not math.isfinite(args.baseline_timeout) or args.baseline_timeout <= 0
    ):
        raise ValueError("baseline_timeout must be finite and positive.")
    if (mode != "mia" or args.compare_smartnoise) and not args.baseline_python.is_file():
        raise FileNotFoundError("Create environments/baselines first and pass its interpreter using --baseline-python.")
    args.baseline_python = args.baseline_python.absolute()
    args.smartnoise_methods = list(dict.fromkeys(args.smartnoise_methods))
    return args


def stream_worker(
    command: list[str], environment: dict[str, str], log_path: Path,
    verbose: int, timeout: float | None = None,
) -> int:
    expired = Event()
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
        ) as process:
            def stop_on_timeout() -> None:
                if process.poll() is None:
                    expired.set()
                    process.kill()

            timer = Timer(timeout, stop_on_timeout) if timeout is not None else None
            if timer is not None:
                timer.daemon = True
                timer.start()
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if verbose:
                        print(line, end="", flush=True)
                returncode = process.wait()
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                if timer is not None:
                    timer.cancel()
            if expired.is_set():
                raise subprocess.TimeoutExpired(command, timeout)
            return returncode


def compare_baselines(
    train_features, train_labels, test_features, test_labels,
    args, context, rows, report, output,
) -> None:
    report["protocol"]["smartnoise_comparison"] = {
        "benchmark_version": "independent_pipelines_v2",
        "space": "SmartNoise uses the original feature dimensions after fixed numeric preprocessing; no DP-TAL encoder is used by baselines.",
        "feature_bins": args.feature_bins, "public_bounds": [-1, 1],
        "training": "Each baseline fits only original training features and labels, once per seed, independently of DP-TAL's noise sweep.",
        "evaluation": "All pipelines use the same real test records and labels. Each classifier trains/tests in its own matching representation.",
        "representations": "DP-TAL uses continuous latents; MWEM/MST/AIM use public-binned original features; DPCTGAN uses continuous original features.",
        "privacy": "Requested SmartNoise budgets belong to independent pipelines, not encoder-plus-SmartNoise composition. Repeated releases still require composition.",
        "budget_matching": "Not a matched-epsilon benchmark; compare DP-TAL accounted epsilon against requested baseline budgets, and inspect unverified legacy accounting warnings.",
        "mwem_partitions": "Public order places label first to avoid a label-only partition; dependencies across partitions are still lost.",
        "limitations": "Quantization can lose signal. Random record splitting has no temporal/device grouping or duplicate removal. MIA is not a privacy proof.",
    }
    comparison_context = {
        "seed": context["seed"], "noise_multiplier": None, "epsilon": None,
        "delta": args.delta, "generator_training_seconds": None,
        "generation_seconds": None, "encoding_seconds": None,
        "encoder_epsilon": None, "baseline_epsilon_requested": args.baseline_epsilon,
        "privacy_budget_verified": False,
    }
    tag = f"seed_{context['seed']}"
    result_path = output / f"smartnoise_{tag}.json"
    config = {
        "methods": args.smartnoise_methods, "epsilon": args.baseline_epsilon, "delta": args.delta,
        "bins": args.feature_bins, "seed": context["seed"], "threads": args.threads,
        "feature_names": context.get("feature_names"), "skip_membership": args.skip_membership,
        "mia_max_records": args.mia_max_records, "mia_target_fpr": args.mia_target_fpr,
        "classifiers": args.classifiers, "classifier_max_iter": args.classifier_max_iter,
        "verbose": args.verbose, "mwem_iterations": args.mwem_iterations,
        "mwem_split_factor": args.mwem_split_factor, "aim_rounds": args.aim_rounds,
        "dpctgan_epochs": args.dpctgan_epochs, "dpctgan_sigma": args.dpctgan_sigma,
        "batch_size": max(2, args.batch_size - args.batch_size % 2),
        "synthetic_samples": args.synthetic_samples,
    }
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    if args.verbose:
        print(f"[SmartNoise] Separate worker | {tag} | Methods: {', '.join(args.smartnoise_methods)}", flush=True)
    with TemporaryDirectory(prefix="smartnoise_input_", dir=output) as directory:
        inputs = Path(directory) / "data.npz"
        request = Path(directory) / "config.json"
        arrays = {"train_features": train_features, "train_labels": train_labels,
                  "test_features": test_features, "test_labels": test_labels}
        membership = context.get("membership_data")
        if membership is not None:
            arrays.update(nonmember_features=membership.nonmember_features, nonmember_labels=membership.nonmember_labels)
        np.savez_compressed(inputs, **arrays)
        request.write_text(json.dumps(config, allow_nan=False), encoding="utf-8")
        command = [str(args.baseline_python), "-u", "-m", "experiments.smartnoise_worker",
                   "--input", str(inputs.resolve()), "--config", str(request.resolve()),
                   "--output", str(result_path.resolve())]
        returncode = stream_worker(
            command, environment, output / f"smartnoise_{tag}.log", args.verbose, args.baseline_timeout,
        )
        if returncode:
            raise RuntimeError(f"SmartNoise worker failed ({returncode}); see smartnoise_{tag}.log.")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    for backend in payload["backends"]:
        report.setdefault("baseline_runs", []).append({**comparison_context, **backend})
        if args.verbose:
            print(f"[SmartNoise] {backend['method']} | Status: {backend['status']}", flush=True)
    for evaluation in payload["evaluations"]:
        parameters = evaluation.pop("classifier_params")
        warnings = evaluation.pop("warnings")
        row = {
            **comparison_context, **evaluation,
            "epsilon": None,
            "baseline_epsilon_requested": args.baseline_epsilon,
            "positive_original_label": args.positive_label,
            "representation": f"smartnoise_{evaluation['method']}",
        }
        rows.append(row)
        report["classifiers"].append({**row, "parameters": parameters, "warnings": warnings})
    for attack in payload.get("membership_inference", []):
        report["membership_inference"].append({
            **comparison_context, **attack, "representation": f"smartnoise_{attack['method']}",
            "positive_original_label": args.positive_label,
        })
    save_results(output, rows, report)


def main() -> None:
    run_experiment(comparison_args(), baseline_comparison=compare_baselines)


if __name__ == "__main__":
    main()