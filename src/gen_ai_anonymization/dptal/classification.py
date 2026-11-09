from time import perf_counter
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from .membership import MembershipConfig, MembershipData, run_membership_attacks


CLASSIFIERS = ("logistic_regression", "mlp")


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, positive_scores: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "precision_positive": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, positive_scores))
            if positive_scores is not None and len(np.unique(labels)) == 2 else None
        ),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "recall_positive": float(true_positive / max(true_positive + false_negative, 1)),
        "false_positive_rate": float(false_positive / max(false_positive + true_negative, 1)),
        "true_negative": int(true_negative), "false_positive": int(false_positive),
        "false_negative": int(false_negative), "true_positive": int(true_positive),
    }


def fit_and_evaluate(
    name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
    max_iter: int = 200,
    verbose: int = 1,
    membership_data: MembershipData | None = None,
    membership_config: MembershipConfig | None = None,
) -> dict:
    if name not in CLASSIFIERS or max_iter < 1:
        raise ValueError("Choose a supported classifier and a positive iteration limit.")
    if name == "logistic_regression":
        classifier = LogisticRegression(max_iter=max_iter, random_state=seed, verbose=int(verbose >= 2))
    else:
        classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", batch_size=min(128, len(train_labels)),
            max_iter=max_iter, random_state=seed, verbose=verbose >= 2,
            early_stopping=False,
        )
    start = perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(train_features, train_labels)
    training_seconds = perf_counter() - start
    messages = [str(item.message) for item in captured]
    if verbose:
        for message in messages:
            print(f"[Classifier warning] {message}", flush=True)
    start = perf_counter()
    predictions = classifier.predict(test_features)
    positive_index = list(classifier.classes_).index(1)
    positive_scores = classifier.predict_proba(test_features)[:, positive_index]
    inference_seconds = perf_counter() - start
    result = {
        **classification_metrics(test_labels, predictions, positive_scores),
        "classifier_training_seconds": training_seconds,
        "classifier_inference_seconds": inference_seconds,
        "converged": not any(issubclass(item.category, ConvergenceWarning) for item in captured),
        "iterations": int(np.max(classifier.n_iter_)),
        "classifier_params": classifier.get_params(), "warnings": messages,
    }
    if membership_data is not None:
        result["membership_inference"] = run_membership_attacks(
            classifier, membership_data, membership_config or MembershipConfig(seed=seed)
        )
    return result