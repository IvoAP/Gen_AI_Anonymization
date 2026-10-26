import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HAS_DP = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas", "diffprivlib"))
HAS_SMARTNOISE = HAS_DP and importlib.util.find_spec("snsynth") is not None

if HAS_DP:
    import numpy as np
    import pandas as pd
    from diffprivlib.accountant import BudgetAccountant
    from diffprivlib.utils import BudgetError

    from anon_algorithms import gaussian_means, laplace_means, synthesize_aim, synthesize_mst


@unittest.skipUnless(HAS_DP, "Install the project dependencies with uv sync.")
class PrivateMeanTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"first": [-5.0, 2.0, np.nan, 20.0], "second": [1.0, 2.0, 3.0, 4.0], "identifier": ["a", "b", "c", "d"]})
        self.bounds = {"first": (0.0, 10.0), "second": (0.0, 4.0)}

    def test_laplace_clips_imputes_splits_budget_and_drops_identifiers(self):
        original = self.data.copy(deep=True)
        accountant = BudgetAccountant(epsilon=2, delta=0)
        with patch("diffprivlib.mechanisms.Laplace") as mechanism:
            mechanism.return_value.randomise.side_effect = lambda value: value
            result = laplace_means(self.data, bounds=self.bounds, epsilon=2, accountant=accountant)
        self.assertEqual(result.to_dict(), {"first": 4.25, "second": 2.5})
        self.assertEqual([call.kwargs["epsilon"] for call in mechanism.call_args_list], [1, 1])
        self.assertEqual([call.kwargs["sensitivity"] for call in mechanism.call_args_list], [2.5, 1])
        self.assertEqual(tuple(accountant.total()), (2, 0))
        pd.testing.assert_frame_equal(self.data, original)

    def test_gaussian_uses_analytic_calibration_and_splits_delta(self):
        accountant = BudgetAccountant(epsilon=4, delta=1e-5)
        with patch("diffprivlib.mechanisms.GaussianAnalytic") as mechanism:
            mechanism.return_value.randomise.side_effect = lambda value: value
            result = gaussian_means(self.data, bounds=self.bounds, epsilon=4, delta=1e-5, accountant=accountant)
        self.assertEqual(result.to_dict(), {"first": 4.25, "second": 2.5})
        self.assertEqual([call.kwargs["epsilon"] for call in mechanism.call_args_list], [2, 2])
        self.assertEqual([call.kwargs["delta"] for call in mechanism.call_args_list], [5e-6, 5e-6])
        self.assertEqual(tuple(accountant.total()), (4, 1e-5))

    def test_real_mechanisms_produce_finite_bounded_outputs(self):
        for mechanism in (laplace_means, gaussian_means):
            with self.subTest(mechanism=mechanism.__name__):
                arguments = {"delta": 1e-6} if mechanism is gaussian_means else {}
                result = mechanism(self.data, bounds=self.bounds, epsilon=2, accountant=BudgetAccountant(), **arguments)
                self.assertTrue(np.isfinite(result).all())
                for column, (lower, upper) in self.bounds.items():
                    self.assertTrue(lower <= result[column] <= upper)

    def test_repeated_releases_cannot_exceed_accountant_budget(self):
        accountant = BudgetAccountant(epsilon=1, delta=0)
        laplace_means(self.data, bounds=self.bounds, epsilon=1, accountant=accountant)
        with self.assertRaises(BudgetError):
            laplace_means(self.data, bounds=self.bounds, epsilon=1, accountant=accountant)

    def test_invalid_epsilon_is_rejected_without_spending_budget(self):
        for epsilon in (0, -1, np.nan, np.inf, True):
            accountant = BudgetAccountant()
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                laplace_means(self.data, bounds=self.bounds, epsilon=epsilon, accountant=accountant)
            self.assertEqual(tuple(accountant.total()), (0, 0))

    def test_invalid_delta_is_rejected(self):
        for delta in (0, -1, 1, np.nan, np.inf, True):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                gaussian_means(self.data, bounds=self.bounds, epsilon=1, delta=delta)

    def test_invalid_schema_is_rejected(self):
        for bounds in ({}, {"first": (1, 1)}, {"first": (5, 1)}, {"first": (0, np.inf)}, {"missing": (0, 1)}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                laplace_means(self.data, bounds=bounds, epsilon=1)

    def test_non_numeric_and_infinite_values_use_public_policy(self):
        data = pd.DataFrame({"value": ["invalid", None, np.inf, -np.inf]})
        with patch("diffprivlib.mechanisms.Laplace") as mechanism:
            mechanism.return_value.randomise.side_effect = lambda value: value
            result = laplace_means(data, bounds={"value": (0, 10)}, epsilon=1)
        self.assertEqual(result["value"], 5)

    def test_empty_data_is_rejected(self):
        with self.assertRaises(ValueError):
            laplace_means(self.data.iloc[:0], bounds=self.bounds, epsilon=1)


@unittest.skipUnless(HAS_SMARTNOISE, "SmartNoise integration tests require uv sync.")
class SmartNoiseTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"label": [0, 1] * 50, "value": [2.0, 8.0] * 50, "identifier": ["private"] * 100})
        self.arguments = {
            "epsilon": 2.0, "delta": 1e-5, "n_rows": 12,
            "categorical_domains": {"label": [0, 1, 2]},
            "numeric_bounds": {"value": (0, 10)}, "bins": 2,
        }

    def test_real_aim_and_mst_return_only_synthetic_schema_columns(self):
        original = self.data.copy(deep=True)
        for algorithm in (synthesize_aim, synthesize_mst):
            with self.subTest(algorithm=algorithm.__name__):
                accountant = BudgetAccountant(epsilon=2, delta=1e-5)
                result = algorithm(self.data, accountant=accountant, **self.arguments)
                self.assertEqual(result.shape, (12, 2))
                self.assertEqual(result.columns.tolist(), ["label", "value"])
                self.assertTrue(result["label"].isin([0, 1, 2]).all())
                self.assertTrue(result["value"].between(0, 10).all())
                self.assertEqual(tuple(accountant.total()), (2, 1e-5))
        pd.testing.assert_frame_equal(self.data, original)

    def test_transformer_is_fit_only_on_public_domains_and_budget_is_calibrated(self):
        self.data.loc[0, "label"] = 99
        with patch("snsynth.Synthesizer.create") as factory:
            factory.return_value.sample.return_value = pd.DataFrame({"label": [0] * 12, "value": [2.5] * 12})
            result = synthesize_mst(self.data, accountant=BudgetAccountant(), **self.arguments)
        transformer = factory.return_value.fit.call_args.kwargs["transformer"]
        prepared = factory.return_value.fit.call_args.args[0]
        self.assertEqual(transformer.cardinality, [3, 2])
        self.assertEqual(prepared.loc[0, "label"], 0)
        self.assertEqual(factory.call_args.kwargs["epsilon"], 1)
        self.assertAlmostEqual(factory.call_args.kwargs["delta"], 1e-5 / (1 + np.exp(1)))
        self.assertEqual(result.shape, (12, 2))

    def test_invalid_domains_and_output_sizes_fail_before_training(self):
        cases = [
            {"n_rows": 0}, {"n_rows": True}, {"bins": 1}, {"delta": 0},
            {"categorical_domains": {"label": [0, 0]}},
            {"categorical_domains": {"label": [0, "one"]}},
            {"categorical_domains": {"label": "01"}},
            {"categorical_domains": {}, "numeric_bounds": {"value": (0, 10)}},
            {"numeric_bounds": {"label": (0, 1)}},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), patch("snsynth.Synthesizer.create") as factory:
                with self.assertRaises(ValueError):
                    synthesize_aim(self.data, **(self.arguments | overrides))
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()