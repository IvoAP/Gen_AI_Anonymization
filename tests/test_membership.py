import unittest

import numpy as np

from gen_ai_anonymization.dptal.membership import (
    MembershipConfig, MembershipData, attack_metrics, calibrated_threshold,
    run_membership_attacks, split_candidates,
)


class ProbabilityClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        return np.column_stack([1 - features[:, 0], features[:, 0]])


class MembershipTests(unittest.TestCase):
    def test_partition_is_disjoint_and_class_matched(self):
        labels = np.tile([0, 1], 40)
        partitions = split_candidates(labels, labels, MembershipConfig(max_records_per_group=40))
        member_calibration, member_evaluation, nonmember_calibration, nonmember_evaluation = partitions
        self.assertEqual(len(np.intersect1d(member_calibration, member_evaluation)), 0)
        self.assertEqual(len(np.intersect1d(nonmember_calibration, nonmember_evaluation)), 0)
        for indices in partitions:
            self.assertEqual(len(indices), 20)
            self.assertEqual(np.bincount(labels[indices]).tolist(), [10, 10])
        repeated = split_candidates(labels, labels, MembershipConfig(max_records_per_group=40))
        for first, second in zip(partitions, repeated):
            np.testing.assert_array_equal(first, second)

    def test_confidence_can_detect_perfect_separation(self):
        result = attack_metrics(np.full(20, 0.1), np.full(20, 0.9), np.full(20, 0.1), 0.01)
        self.assertEqual(result["roc_auc"], 1)
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["tpr"], 1)
        self.assertEqual(result["fpr"], 0)

    def test_equal_confidence_is_chance_not_perfect_attack(self):
        result = attack_metrics(np.ones(20), np.ones(20), np.ones(20), 0.01)
        self.assertEqual(result["roc_auc"], 0.5)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["advantage"], 0)

    def test_threshold_uses_only_calibration_nonmembers_and_handles_ties(self):
        calibration = np.array([0.1, 0.2, 0.2, 0.9])
        threshold = calibrated_threshold(calibration, 0.25)
        self.assertLessEqual(np.mean(calibration > threshold), 0.25)
        result = attack_metrics(calibration, np.ones(10), np.ones(10), 0.25)
        self.assertEqual(result["threshold"], threshold)
        self.assertGreater(result["fpr"], 0.25)

    def test_both_attacks_query_real_candidates(self):
        labels = np.tile([0, 1], 40)
        member_probability = np.where(labels == 1, 0.99, 0.01)[:, None]
        nonmember_probability = np.where(labels == 1, 0.6, 0.4)[:, None]
        data = MembershipData(member_probability, labels, nonmember_probability, labels, "generator_training_records")
        results = run_membership_attacks(ProbabilityClassifier(), data, MembershipConfig())
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["membership_scope"], "generator_training_records")
            self.assertEqual(result["roc_auc"], 1)
            self.assertEqual(result["evaluation_members"], result["evaluation_nonmembers"])

    def test_insufficient_candidates_are_reported(self):
        labels = np.array([0, 1])
        data = MembershipData(np.ones((2, 1)), labels, np.ones((2, 1)), labels, "classifier_training_records")
        results = run_membership_attacks(ProbabilityClassifier(), data, MembershipConfig())
        self.assertEqual(results[0]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()