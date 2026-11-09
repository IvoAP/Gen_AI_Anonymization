import unittest
from unittest.mock import MagicMock, patch
import warnings

import pandas as pd

from gen_ai_anonymization.baseline.base import SmartNoiseBaseline
from gen_ai_anonymization.baseline import AIMBaseline, DPCTGANBaseline, MSTBaseline, MWEMBaseline


class BaselineInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"feature": [0, 1, 0, 1], "label": [0, 0, 1, 1]})

    def test_invalid_privacy_budgets(self):
        for epsilon, preprocessing in ((0, 0), (-1, 0), (float("nan"), 0), (1, 1), (1, -1)):
            with self.subTest(epsilon=epsilon, preprocessing=preprocessing):
                with self.assertRaises(ValueError):
                    SmartNoiseBaseline("mwem", epsilon, preprocessor_eps=preprocessing)

    def test_individual_modules_select_the_correct_backend(self):
        for factory, name in ((MWEMBaseline, "mwem"), (MSTBaseline, "mst"),
                              (AIMBaseline, "aim"), (DPCTGANBaseline, "dpctgan")):
            with self.subTest(method=name):
                baseline = factory(epsilon=3, preprocessor_eps=0.5)
                backend = MagicMock()
                backend.sample.return_value = self.data.iloc[:2]
                with patch.object(baseline, "_create_synthesizer", return_value=backend):
                    baseline.fit_sample(self.data, 2, categorical_columns=list(self.data.columns))
                self.assertEqual(baseline.method, name)
                self.assertEqual(baseline.get_report()["epsilon_requested"], 3)
                self.assertEqual(baseline.get_report()["preprocessor_epsilon_allocated"], 0.5)

    def test_dpctgan_rejects_nonprivate_and_invalid_options(self):
        for options in ({"disabled_dp": True}, {"sigma": 0}, {"batch_size": 3},
                        {"max_per_sample_grad_norm": 0}, {"delta": 1}, {"pac": 2}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    DPCTGANBaseline(3, **options)

    def test_factory_passes_parameters_and_enforces_environment(self):
        factory = MagicMock()
        fake_module = MagicMock()
        fake_module.Synthesizer.create = factory
        with patch.dict("sys.modules", {"snsynth": fake_module}):
            with patch("gen_ai_anonymization.baseline.base.version", return_value="0.14.0"):
                baseline = DPCTGANBaseline(3, epochs=1, batch_size=4, sigma=2)
                baseline._create_synthesizer()
                factory.assert_called_once_with("dpctgan", epsilon=3.0, **baseline.parameters)
            with patch("gen_ai_anonymization.baseline.base.version", return_value="1.5.0"):
                with self.assertRaises(ImportError):
                    MWEMBaseline(3)._create_synthesizer()

    def test_fit_sample_delegates_budget_and_column_hints(self):
        backend = MagicMock()
        backend.sample.return_value = self.data.iloc[:2]
        baseline = SmartNoiseBaseline("mwem", 3, preprocessor_eps=0.5)
        with patch.object(baseline, "_create_synthesizer", return_value=backend):
            samples = baseline.fit_sample(self.data, 2, categorical_columns=["label"], ordinal_columns=["feature"])
        self.assertEqual(samples.shape, (2, 2))
        backend.fit.assert_called_once_with(
            self.data, transformer=None, categorical_columns=["label"], ordinal_columns=["feature"],
            continuous_columns=[], preprocessor_eps=0.5, nullable=False,
        )
        backend.sample.assert_called_once_with(2)
        self.assertEqual(baseline.get_report()["epsilon_requested"], 3)
        self.assertIsNotNone(baseline.training_seconds)

    def test_requires_fit_and_valid_samples(self):
        baseline = SmartNoiseBaseline("mst", 1)
        with self.assertRaises(RuntimeError):
            baseline.sample(2)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                baseline.fit_sample(self.data, count)

    def test_bad_column_hints_fail_before_backend_creation(self):
        baseline = SmartNoiseBaseline("aim", 1)
        with self.assertRaises(ValueError):
            baseline.fit(self.data, categorical_columns=["missing"])
        with self.assertRaises(ValueError):
            baseline.fit(self.data, categorical_columns=["label"], ordinal_columns=["label"])

    def test_failed_refit_does_not_sample_previous_model(self):
        backend = MagicMock()
        baseline = SmartNoiseBaseline("mwem", 1)
        with patch.object(baseline, "_create_synthesizer", return_value=backend):
            baseline.fit(self.data)
            backend.fit.side_effect = ValueError("Backend failed")
            with self.assertRaises(ValueError):
                baseline.fit(self.data)
        with self.assertRaises(RuntimeError):
            baseline.sample(2)

    def test_backend_privacy_warnings_are_preserved(self):
        backend = MagicMock()
        backend.fit.side_effect = lambda *args, **kwargs: warnings.warn("Privacy budget requires review", UserWarning)
        baseline = DPCTGANBaseline(3)
        with patch.object(baseline, "_create_synthesizer", return_value=backend):
            with self.assertWarns(UserWarning):
                baseline.fit(self.data)
        self.assertEqual(baseline.get_report()["backend_warnings"], ["Privacy budget requires review"])
        self.assertFalse(baseline.get_report()["privacy_budget_verified"])


if __name__ == "__main__":
    unittest.main()