from pathlib import Path
from importlib.util import find_spec
from tempfile import TemporaryDirectory
import unittest
import sys
from contextlib import redirect_stdout
import io
import os
import subprocess
from unittest.mock import MagicMock, patch

import numpy as np

from experiments.common import DEFAULT_DATASET, parse_args
from experiments.smartnoise import comparison_args, quantize_features, stream_worker


class ExperimentTests(unittest.TestCase):
    def test_default_dataset_survives_executor_move(self):
        self.assertEqual(DEFAULT_DATASET, Path(__file__).resolve().parents[1] / "data" / "ddos_csv_100000.csv")
        self.assertTrue(DEFAULT_DATASET.is_file())

    def test_modes_are_independent(self):
        self.assertTrue(parse_args([], mode="dptal").skip_membership)
        self.assertTrue(parse_args([], mode="smartnoise").skip_membership)
        self.assertFalse(parse_args([], mode="mia").skip_membership)
        self.assertEqual(parse_args([], mode="mia").output, Path("results/mia"))

    def test_baseline_interpreter_preserves_virtualenv_symlink(self):
        args = comparison_args(["--baseline-python", sys.executable])
        self.assertEqual(args.baseline_python, Path(sys.executable).absolute())

    def test_dpctgan_is_enabled_by_default_and_can_be_selected_alone(self):
        args = comparison_args(["--baseline-python", sys.executable])
        self.assertEqual(args.smartnoise_methods, ["mwem", "mst", "aim", "dpctgan"])
        explicit = comparison_args(["--baseline-python", sys.executable, "--smartnoise-methods", "dpctgan"])
        self.assertEqual(explicit.smartnoise_methods, ["dpctgan"])

    def test_aim_rejects_rounds_that_exhaust_initial_budget(self):
        import pandas as pd
        from gen_ai_anonymization.baseline import AIMBaseline

        with self.assertRaisesRegex(ValueError, "initial marginals"):
            AIMBaseline(3, rounds=2).fit(pd.DataFrame({"a": [0, 1], "b": [0, 1], "label": [0, 1]}))

    def test_quantization_is_public_bounded_and_consistent(self):
        values = np.array([[-100, -4, -2, 0, 2, 4, 100]], dtype=np.float32)
        np.testing.assert_array_equal(quantize_features(values, 4, 4), [[0, 0, 1, 2, 3, 3, 3]])
        np.testing.assert_array_equal(quantize_features(values[:, 2:5], 4, 4), [[1, 2, 3]])

    def test_quantization_rejects_invalid_inputs(self):
        for bins, bound in ((1, 4), (4, 0), (4, float("inf"))):
            with self.assertRaises(ValueError):
                quantize_features(np.zeros((2, 2)), bins, bound)
        with self.assertRaises(ValueError):
            quantize_features(np.array([[np.nan]]), 4, 4)

    def test_worker_output_is_flushed_before_waiting_for_exit(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.stdout = iter(["Training started\n", "Backend warning\n"])
        captured = io.StringIO()
        with TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"

            def verify_output():
                self.assertIn("Training started", log.read_text(encoding="utf-8"))
                self.assertIn("Training started", captured.getvalue())
                return 0

            process.wait.side_effect = verify_output
            with patch("experiments.smartnoise.subprocess.Popen", return_value=process) as launch:
                with redirect_stdout(captured):
                    self.assertEqual(stream_worker(["python", "-u"], {}, log, verbose=1), 0)
            self.assertEqual(launch.call_args.kwargs["stderr"], subprocess.STDOUT)
            self.assertIn("Backend warning", log.read_text(encoding="utf-8"))

    def test_real_worker_output_is_saved_in_quiet_mode(self):
        with TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            command = [sys.executable, "-u", "-c", "import sys; print('started'); print('warning', file=sys.stderr)"]
            captured = io.StringIO()
            with redirect_stdout(captured):
                self.assertEqual(stream_worker(command, dict(os.environ), log, verbose=0), 0)
            self.assertEqual(captured.getvalue(), "")
            self.assertIn("started", log.read_text(encoding="utf-8"))
            self.assertIn("warning", log.read_text(encoding="utf-8"))

    def test_silent_worker_timeout_still_terminates_process(self):
        with TemporaryDirectory() as directory:
            command = [sys.executable, "-u", "-c", "exec('while True: pass')"]
            with self.assertRaises(subprocess.TimeoutExpired):
                stream_worker(command, dict(os.environ), Path(directory) / "worker.log", verbose=0, timeout=0.2)


@unittest.skipUnless(find_spec("snsynth") is not None, "Requires the separate SmartNoise environment.")
class SmartNoiseWorkerTests(unittest.TestCase):
    def test_real_backends_use_public_domains_and_shared_metrics(self):
        from experiments.smartnoise_worker import run_worker

        generator = np.random.default_rng(42)
        features = generator.uniform(-1, 1, size=(160, 2)).astype(np.float32)
        labels = (features[:, 0] >= 0).astype(np.int64)
        config = {
            "methods": ["mwem", "mst", "aim", "dpctgan"], "epsilon": 10, "delta": 1e-6,
            "bins": 4, "seed": 42, "threads": 1, "classifiers": ["logistic_regression"],
            "classifier_max_iter": 20, "verbose": 1, "mwem_iterations": 2,
            "mwem_split_factor": 2, "aim_rounds": 4, "dpctgan_epochs": 1,
            "dpctgan_sigma": 5, "batch_size": 16, "synthetic_samples": 128,
        }
        with TemporaryDirectory() as directory:
            inputs = Path(directory) / "data.npz"
            np.savez_compressed(inputs, train_features=features[:128], train_labels=labels[:128],
                                test_features=features[128:], test_labels=labels[128:])
            captured = io.StringIO()
            with redirect_stdout(captured):
                report = run_worker(inputs, config, Path(directory) / "results.json")
        self.assertEqual([item["status"] for item in report["backends"]], ["completed"] * 4, report["backends"])
        self.assertEqual(len(report["evaluations"]), 4)
        logs = captured.getvalue()
        for method in config["methods"]:
            self.assertIn(f"[SmartNoise:{method}] START", logs)
            self.assertIn(f"[SmartNoise:{method}] Training started", logs)
            self.assertIn(f"[SmartNoise:{method}] Training completed", logs)
            self.assertIn(f"[SmartNoise:{method}] Sampling completed", logs)
            self.assertIn(f"[SmartNoise:{method}] Classifier logistic_regression completed", logs)
            self.assertIn(f"[SmartNoise:{method}] END | Status: completed", logs)
        for evaluation in report["evaluations"]:
            self.assertEqual(evaluation["test_records"], 32)
            self.assertEqual(evaluation["train_records"], 128)
            for metric in ("accuracy", "f1", "precision_positive", "recall_positive", "roc_auc", "cohen_kappa"):
                self.assertIn(metric, evaluation)