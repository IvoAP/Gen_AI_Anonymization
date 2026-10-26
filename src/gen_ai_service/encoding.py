import numpy as np
import pandas as pd


class TabularEncoder:
    """Fit training-only scaling and bounded-size one-hot categorical encoding."""

    def fit(self, data: pd.DataFrame, max_encoded_features: int = 4096) -> "TabularEncoder":
        self.columns = data.columns.copy()
        self.dtypes = data.dtypes
        self.num_cols = data.select_dtypes(include="number").columns.tolist()
        self.cat_cols = [column for column in data.columns if column not in self.num_cols]
        if any(pd.api.types.is_complex_dtype(data[column].dtype) for column in self.num_cols):
            raise ValueError("Complex numeric columns are not supported.")
        numeric = data[self.num_cols].astype("float64").replace([np.inf, -np.inf], np.nan)
        self.mean = numeric.mean().fillna(0.0)
        self.std = numeric.std(ddof=0).replace(0, 1).fillna(1.0)
        self.lower = numeric.min().fillna(0.0)
        self.upper = numeric.max().fillna(0.0)
        self.categories = {}
        self.missing_tokens = {}
        for column in self.cat_cols:
            values = data[column].dropna().astype(object).unique().tolist()
            token = "__MISSING__"
            while token in values:
                token += "_"
            self.missing_tokens[column] = token
            self.categories[column] = values + [token]
        self.input_dim = len(self.num_cols) + sum(self.cat_sizes)
        if self.input_dim > max_encoded_features:
            raise ValueError(
                f"Encoding needs {self.input_dim} features; limit is {max_encoded_features}. "
                "Select fewer columns or explicitly preprocess high-cardinality fields."
            )
        return self

    @property
    def cat_sizes(self) -> list[int]:
        return [len(self.categories[column]) for column in self.cat_cols]

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        numeric = data[self.num_cols].astype("float64").replace([np.inf, -np.inf], np.nan)
        numeric = (numeric.fillna(self.mean) - self.mean) / self.std
        parts = [numeric.to_numpy(dtype=np.float32)]
        for column in self.cat_cols:
            values = data[column].astype(object).where(data[column].notna(), self.missing_tokens[column])
            codes = pd.Categorical(values, categories=self.categories[column]).codes.copy()
            codes[codes < 0] = len(self.categories[column]) - 1
            encoded = np.zeros((len(data), len(self.categories[column])), dtype=np.float32)
            encoded[np.arange(len(data)), codes] = 1.0
            parts.append(encoded)
        encoded = np.concatenate(parts, axis=1)
        if not np.isfinite(encoded).all():
            raise ValueError("Encoding produced non-finite values; rescale extreme numeric columns first.")
        return encoded

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        numeric_count = len(self.num_cols)
        numeric = pd.DataFrame(encoded[:, :numeric_count], columns=self.num_cols)
        numeric = (numeric * self.std + self.mean).clip(self.lower, self.upper, axis=1)
        for column in self.num_cols:
            if pd.api.types.is_integer_dtype(self.dtypes[column]):
                numeric[column] = numeric[column].round().astype(self.dtypes[column])
        result = numeric
        start = numeric_count
        for column, size in zip(self.cat_cols, self.cat_sizes):
            codes = encoded[:, start:start + size].argmax(axis=1)
            categories = np.array(self.categories[column], dtype=object)
            values = pd.Series(categories[codes])
            result[column] = values.mask(values == self.missing_tokens[column], np.nan)
            start += size
        return result.loc[:, self.columns]