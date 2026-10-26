from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gen_ai_service import Anonymizer, AutoencoderConfig, anonymize_dataset
from gen_ai_service.anonymize import TabularEncoder, add_latent_noise


class EncoderTests(unittest.TestCase):
    def test_missing_constant_and_unknown_values_are_finite(self):
        data = pd.DataFrame({"constant": [3.0, 3.0, np.nan], "empty": [np.nan] * 3, "category": ["a", None, "b"]})
        encoder = TabularEncoder().fit(data)
        encoded = encoder.transform(data)
        self.assertTrue(np.isfinite(encoded).all())
        transformed = encoder.transform(pd.DataFrame({"constant": [3.0], "empty": [np.nan], "category": ["new"]}))
        restored = encoder.inverse_transform(transformed)
        self.assertTrue(pd.isna(restored.loc[0, "category"]))
        self.assertEqual(restored.loc[0, "constant"], 3.0)

    def test_float_precision_and_integer_columns_are_preserved(self):
        data = pd.DataFrame({"ratio": [0.125, 0.375, 0.625], "count": [1, 2, 3]})
        encoder = TabularEncoder().fit(data)
        restored = encoder.inverse_transform(encoder.transform(data))
        np.testing.assert_allclose(restored["ratio"], data["ratio"], atol=1e-6)
        pd.testing.assert_series_equal(restored["count"], data["count"])

    def test_categorical_dtype_and_missing_token_collision(self):
        data = pd.DataFrame({"kind": pd.Categorical(["__MISSING__", "normal", None])})
        encoder = TabularEncoder().fit(data)
        restored = encoder.inverse_transform(encoder.transform(data))
        self.assertEqual(restored.loc[0, "kind"], "__MISSING__")
        self.assertTrue(pd.isna(restored.loc[2, "kind"]))

    def test_high_cardinality_is_rejected_before_dense_encoding(self):
        with self.assertRaisesRegex(ValueError, "Encoding needs"):
            TabularEncoder().fit(pd.DataFrame({"identifier": list("abcdefgh")}), max_encoded_features=4)


class AutoencoderTests(unittest.TestCase):
    def setUp(self):
        self.config = AutoencoderConfig(hidden=(8, 4), latent_dim=2, max_epochs=2, patience=1, batch_size=7)
        self.data = pd.DataFrame({
            "ratio": np.linspace(0.1, 0.9, 20),
            "constant": [4.0] * 20,
            "Attack_type": ["Normal", "Attack"] * 10,
        }, index=pd.Index(range(100, 120), name="record"))

    def test_training_and_generation_preserve_rows_schema_and_input(self):
        original = self.data.copy(deep=True)
        model = Anonymizer(self.data, config=self.config)
        generated = model.generate()
        self.assertEqual(generated.shape, self.data.shape)
        self.assertTrue(generated.index.equals(self.data.index))
        self.assertTrue(generated.columns.equals(self.data.columns))
        self.assertTrue(np.isfinite(generated.select_dtypes(include="number")).all().all())
        self.assertTrue(generated["ratio"].between(model.encoder.lower["ratio"], model.encoder.upper["ratio"]).all())
        pd.testing.assert_frame_equal(self.data, original)
        pd.testing.assert_frame_equal(generated, model.generate())
        self.assertTrue(1 <= model.best_epoch <= self.config.max_epochs)
        self.assertTrue(np.isfinite(model.history[["train_loss", "val_loss"]]).all().all())

    def test_preprocessing_is_fitted_only_on_training_rows(self):
        permutation = torch.randperm(len(self.data), generator=torch.Generator().manual_seed(42))
        validation_size = max(1, int(len(self.data) * self.config.val_fraction))
        model = Anonymizer(self.data, config=self.config)
        expected_mean = self.data.iloc[permutation[validation_size:].numpy()]["ratio"].mean()
        self.assertAlmostEqual(model.encoder.mean["ratio"], expected_mean)

    def test_numeric_only_and_categorical_only_training(self):
        for data in (self.data[["ratio"]], self.data[["Attack_type"]]):
            with self.subTest(columns=data.columns.tolist()):
                result = anonymize_dataset(data, config=self.config)
                self.assertEqual(result.shape, data.shape)

    def test_global_rng_is_not_changed(self):
        state = torch.random.get_rng_state().clone()
        model = Anonymizer(self.data, config=self.config)
        model.generate()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_training_noise_and_single_row_batches_are_finite(self):
        config = replace(self.config, train_noise=0.2, batch_size=1, max_epochs=1)
        model = Anonymizer(self.data.iloc[:3], config=config)
        self.assertTrue(np.isfinite(model.generate()["ratio"]).all())

    def test_latent_noise_uses_given_spread_and_seed(self):
        latent = torch.zeros(3, 2)
        spread = torch.tensor([1.0, 2.0])
        actual = add_latent_noise(latent, 0.5, torch.Generator().manual_seed(7), spread)
        expected = torch.randn((3, 2), generator=torch.Generator().manual_seed(7)) * spread * 0.5
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(add_latent_noise(latent, 0), latent))
        self.assertTrue(torch.isfinite(add_latent_noise(torch.ones(1, 2), 1)).all())
        for scale in (-1, np.nan, np.inf, True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                add_latent_noise(latent, scale)

    def test_invalid_data_and_configuration_are_rejected(self):
        for data in (self.data.iloc[:0], self.data.iloc[:1], pd.DataFrame(index=[0, 1]), pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])):
            with self.subTest(shape=data.shape), self.assertRaises(ValueError):
                Anonymizer(data, config=self.config)
        for arguments in ({"max_epochs": 0}, {"val_fraction": 1}, {"dropout": 1}, {"train_noise": -1}, {"hidden": ()}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                AutoencoderConfig(**arguments)


if __name__ == "__main__":
    unittest.main()