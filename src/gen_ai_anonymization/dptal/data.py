from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetSplit:
    features: np.ndarray
    labels: np.ndarray
    train: np.ndarray
    attack_holdout: np.ndarray
    test: np.ndarray
    feature_names: list[str]


def load_dataset(path: Path, label_column: str, max_rows: int | None, seed: int) -> DatasetSplit:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    if label_column not in frame or frame.columns.duplicated().any():
        raise ValueError("A unique label column and unique feature names are required.")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(frame))
    if max_rows is not None:
        if max_rows < 100:
            raise ValueError("max_rows must be at least 100.")
        permutation = permutation[:max_rows]
    frame = frame.iloc[permutation].reset_index(drop=True)
    labels = pd.to_numeric(frame[label_column], errors="raise").to_numpy()
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Labels must contain both 0 and 1. Verify which label represents an attack.")
    raw_features = frame.drop(columns=label_column)
    if raw_features.shape[1] == 0 or len(frame) < 100:
        raise ValueError("At least one feature and 100 records are required.")
    numeric = raw_features.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(np.sign(numeric) * np.log1p(np.abs(numeric)) / 20.0, -1, 1).astype(np.float32)
    train_end, holdout_end = int(len(frame) * 0.6), int(len(frame) * 0.8)
    indices = np.arange(len(frame))
    split = DatasetSplit(
        features, labels.astype(np.int64), indices[:train_end],
        indices[train_end:holdout_end], indices[holdout_end:], list(raw_features.columns)
    )
    for name in ("train", "attack_holdout", "test"):
        if len(np.unique(split.labels[getattr(split, name)])) != 2:
            raise ValueError(f"Both classes are required in the {name} partition; use more records.")
    return split