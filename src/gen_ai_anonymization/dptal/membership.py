from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MembershipData:
    member_features: np.ndarray
    member_labels: np.ndarray
    nonmember_features: np.ndarray
    nonmember_labels: np.ndarray
    membership_scope: str


@dataclass(frozen=True)
class MembershipConfig:
    max_records_per_group: int = 2000
    target_fpr: float = 0.01
    seed: int = 42


def split_candidates(
    member_labels: np.ndarray, nonmember_labels: np.ndarray, config: MembershipConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if config.max_records_per_group < 8 or not 0 < config.target_fpr < 1:
        raise ValueError("Membership attacks require at least eight candidates per group and FPR in (0, 1).")
    if not np.isin(member_labels, [0, 1]).all() or not np.isin(nonmember_labels, [0, 1]).all():
        raise ValueError("Membership candidates require binary task labels.")
    members = [np.flatnonzero(member_labels == label) for label in (0, 1)]
    nonmembers = [np.flatnonzero(nonmember_labels == label) for label in (0, 1)]
    count = min(config.max_records_per_group // 2, *(len(indices) for indices in members + nonmembers))
    if count < 4:
        raise ValueError("At least four member and nonmember records per task class are required.")
    rng = np.random.default_rng(config.seed)
    members = [rng.permutation(indices)[:count] for indices in members]
    nonmembers = [rng.permutation(indices)[:count] for indices in nonmembers]
    midpoint = count // 2
    return (
        np.concatenate([indices[:midpoint] for indices in members]),
        np.concatenate([indices[midpoint:] for indices in members]),
        np.concatenate([indices[:midpoint] for indices in nonmembers]),
        np.concatenate([indices[midpoint:] for indices in nonmembers]),
    )


def probability_features(classifier, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float64)
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise ValueError("Membership attacks require finite class probabilities.")
    class_columns = {label: index for index, label in enumerate(classifier.classes_)}
    columns = np.array([class_columns[label] for label in labels])
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    true_probability = probabilities[np.arange(len(labels)), columns]
    loss = -np.log(true_probability)
    entropy = -(probabilities * np.log(probabilities)).sum(axis=1)
    return np.column_stack([loss, entropy])


def calibrated_threshold(nonmember_scores: np.ndarray, target_fpr: float) -> float:
    allowed_false_positives = int(np.floor(target_fpr * len(nonmember_scores)))
    return float(np.sort(nonmember_scores)[::-1][allowed_false_positives])


def attack_metrics(
    calibration_nonmember_scores: np.ndarray, member_scores: np.ndarray,
    nonmember_scores: np.ndarray, target_fpr: float,
) -> dict:
    threshold = calibrated_threshold(calibration_nonmember_scores, target_fpr)
    scores = np.concatenate([member_scores, nonmember_scores])
    membership = np.concatenate([np.ones(len(member_scores), dtype=int), np.zeros(len(nonmember_scores), dtype=int)])
    predictions = (scores > threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        membership, predictions, labels=[0, 1]
    ).ravel()
    tpr = float(true_positive / (true_positive + false_negative))
    fpr = float(false_positive / (false_positive + true_negative))
    return {
        "roc_auc": float(roc_auc_score(membership, scores)),
        "accuracy": float(accuracy_score(membership, predictions)),
        "f1": float(f1_score(membership, predictions, zero_division=0)),
        "tpr": tpr, "fpr": fpr, "advantage": tpr - fpr,
        "threshold": threshold, "target_calibration_fpr": target_fpr,
        "evaluation_members": len(member_scores), "evaluation_nonmembers": len(nonmember_scores),
        "fpr_resolution": 1 / len(nonmember_scores),
        "true_positive": int(true_positive), "false_positive": int(false_positive),
        "true_negative": int(true_negative), "false_negative": int(false_negative),
    }


def run_membership_attacks(classifier, data: MembershipData, config: MembershipConfig) -> list[dict]:
    start = perf_counter()
    try:
        member_calibration, member_evaluation, nonmember_calibration, nonmember_evaluation = split_candidates(
            data.member_labels, data.nonmember_labels, config
        )
    except ValueError as error:
        return [{"status": "skipped", "membership_scope": data.membership_scope, "reason": str(error)}]
    calibration_member_features = probability_features(
        classifier, data.member_features[member_calibration], data.member_labels[member_calibration]
    )
    calibration_nonmember_features = probability_features(
        classifier, data.nonmember_features[nonmember_calibration], data.nonmember_labels[nonmember_calibration]
    )
    evaluation_member_features = probability_features(
        classifier, data.member_features[member_evaluation], data.member_labels[member_evaluation]
    )
    evaluation_nonmember_features = probability_features(
        classifier, data.nonmember_features[nonmember_evaluation], data.nonmember_labels[nonmember_evaluation]
    )
    query_seconds = perf_counter() - start
    metadata = {
        "status": "completed", "membership_scope": data.membership_scope,
        "calibration_members": len(member_calibration), "calibration_nonmembers": len(nonmember_calibration),
        "query_seconds": query_seconds,
        "threat_model": "Probability access and known task label; supervised calibration membership is known.",
        "low_fpr_sample_warning": len(nonmember_evaluation) * config.target_fpr < 10,
    }
    start = perf_counter()
    confidence = attack_metrics(
        -calibration_nonmember_features[:, 0], -evaluation_member_features[:, 0],
        -evaluation_nonmember_features[:, 0], config.target_fpr,
    )
    results = [{**metadata, "attack": "true_label_confidence_threshold", **confidence,
                "attack_seconds": perf_counter() - start}]
    start = perf_counter()
    attack_train = np.concatenate([calibration_member_features, calibration_nonmember_features])
    attack_labels = np.concatenate([
        np.ones(len(member_calibration), dtype=int), np.zeros(len(nonmember_calibration), dtype=int)
    ])
    attacker = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=config.seed))
    attacker.fit(attack_train, attack_labels)
    learned = attack_metrics(
        attacker.predict_proba(calibration_nonmember_features)[:, 1],
        attacker.predict_proba(evaluation_member_features)[:, 1],
        attacker.predict_proba(evaluation_nonmember_features)[:, 1], config.target_fpr,
    )
    results.append({**metadata, "attack": "logistic_loss_entropy", **learned,
                    "attack_seconds": perf_counter() - start})
    return results