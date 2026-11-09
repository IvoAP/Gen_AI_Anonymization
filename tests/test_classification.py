import unittest

import numpy as np

from gen_ai_anonymization.dptal.classification import classification_metrics, fit_and_evaluate


class ClassificationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 0, 1, 1])

    def test_perfect_predictions(self):
        metrics = classification_metrics(self.labels, self.labels, np.array([0.1, 0.2, 0.8, 0.9]))
        for name in ("accuracy", "f1", "macro_f1", "precision_positive", "recall_positive", "roc_auc", "cohen_kappa"):
            self.assertEqual(metrics[name], 1.0, name)
        self.assertEqual(metrics["false_positive_rate"], 0.0)

    def test_all_positive_predictions(self):
        metrics = classification_metrics(self.labels, np.ones(4, dtype=int), np.full(4, 0.5))
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)
        self.assertEqual(metrics["precision_positive"], 0.5)
        self.assertEqual(metrics["recall_positive"], 1.0)
        self.assertEqual(metrics["false_positive_rate"], 1.0)
        self.assertEqual(metrics["roc_auc"], 0.5)

    def test_auc_uses_scores_not_hard_predictions(self):
        metrics = classification_metrics(self.labels, np.ones(4, dtype=int), np.array([0.6, 0.7, 0.8, 0.9]))
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_no_positive_predictions(self):
        metrics = classification_metrics(self.labels, np.zeros(4, dtype=int), np.zeros(4))
        self.assertEqual(metrics["precision_positive"], 0.0)
        self.assertEqual(metrics["recall_positive"], 0.0)
        self.assertEqual(metrics["false_positive_rate"], 0.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)

    def test_auc_is_unavailable_without_scores_or_both_classes(self):
        self.assertIsNone(classification_metrics(self.labels, self.labels)["roc_auc"])
        metrics = classification_metrics(np.ones(4, dtype=int), np.zeros(4, dtype=int), np.zeros(4))
        self.assertIsNone(metrics["roc_auc"])

    def test_both_classifiers_report_new_metrics(self):
        features = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        for name in ("logistic_regression", "mlp"):
            with self.subTest(classifier=name):
                result = fit_and_evaluate(name, features, self.labels, features, self.labels, 42, max_iter=10, verbose=0)
                for metric in ("accuracy", "macro_f1", "precision_positive", "recall_positive", "false_positive_rate", "roc_auc"):
                    self.assertTrue(np.isfinite(result[metric]), metric)
                    self.assertGreaterEqual(result[metric], 0.0)
                    self.assertLessEqual(result[metric], 1.0)


if __name__ == "__main__":
    unittest.main()