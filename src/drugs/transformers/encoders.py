import re
from collections import OrderedDict
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

from drugs.utils.constants import (
    ALL_LABELS,
    BINARY_COLUMNS,
    DESCRIPTION_COLUMN,
    HIGH_CARD_COLUMNS,
    ONE_HOT_COLUMNS,
    PRICE,
    REIMBURSEMENT_RATE,
    SEED,
    STRS_TO_CHECK,
)

np.random.seed(SEED)


class DescriptionEncoder(BaseEstimator, TransformerMixin):
    """
    Feature extraction from description
    """

    def __init__(self, labels: List[str] = None):
        self._labels = ALL_LABELS if labels is None else labels
        self._pattern = "\d+\.*\d* [a-z]+"

    @property
    def labels(self) -> List[str]:
        return self._labels

    def match(self, description: str) -> Tuple[Any, ...]:
        """
        Find labels in description and their count
        """
        ret = OrderedDict((label, 0) for label in self._labels)
        matches = re.findall(self._pattern, description)
        for match in matches:
            quantity, label = match.split()
            if label in self._labels:
                ret[label] = quantity
        return tuple(ret.values())

    def fit(self, x: pd.DataFrame = None, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        f_cols = [col + "_feature" for col in self._labels]
        x_copy = x.copy()
        x_copy[f_cols] = pd.DataFrame(
            x_copy[DESCRIPTION_COLUMN].map(self.match).tolist()
        )
        x_copy[f_cols] = x_copy[f_cols].astype(float)
        return x_copy

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None, **fit_params):
        return self.fit(x, y).transform(x, y)


class PercentageEncoder(BaseEstimator, TransformerMixin):
    """
    Encode percentages as float between 0 and 1
    """

    def __init__(self, column: str = REIMBURSEMENT_RATE):
        self.column = column

    def fit(self, x: pd.DataFrame = None, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        x_copy = x.copy()
        x_copy[self.column + "_feature"] = (
            x_copy[self.column].str.replace("%", "").astype(int) / 100
        )

        return x_copy

    def fit_transform(self, x: pd.DataFrame, y=None, **fit_params):
        return self.fit(x, y).transform(x, y)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Encode high cardinality columns transformers with a mean value of the price
    """

    def __init__(
        self,
        columns: List[str] = None,
        noise_level: float = 0.01,
        min_samples_leaf: int = 1,
        smoothing: int = 6,
    ):
        self.columns = HIGH_CARD_COLUMNS if columns is None else columns
        self.noise_level = noise_level
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self._columns_price_map = {}
        self._prior = 0.00

    def _add_noise(self, series: pd.Series) -> pd.Series:
        return series * (1 + self.noise_level * np.random.randn(len(series)))

    def _smooth(self, series: pd.Series):
        return 1 / (1 + np.exp(-(series - self.min_samples_leaf) / self.smoothing))

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        x_copy = x.copy()
        x_copy[PRICE] = y
        self._prior = y.mean()

        for col in self.columns:
            avgs = x_copy.groupby(col, sort=False)[PRICE].agg(["mean", "count"])
            smooths = self._smooth(avgs["count"])

            col_map = self._prior * (1 - smooths) + avgs["mean"] * smooths
            self._columns_price_map[col] = self._add_noise(col_map)
        return self

    def transform(self, df: pd.DataFrame, y: pd.Series = None):
        df_copy = df.copy()
        for col in self.columns:
            df_copy[col + "_feature"] = (
                df_copy[col].map(self._columns_price_map[col]).fillna(self._prior)
            )
        return df_copy

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None, **fit_params):
        return self.fit(x, y).transform(x, y)


class TargetEncoderCV(BaseEstimator, TransformerMixin):
    """
    Cross-fold target encoder to prevent over fitting
    """

    def __init__(
        self,
        columns: List[str] = None,
        n_splits=3,
        noise_level: float = 0.01,
        min_samples_leaf: int = 1,
        smoothing: int = 6,
    ):
        self.columns = HIGH_CARD_COLUMNS if columns is None else columns
        self.n_splits = n_splits
        self.noise_level = noise_level
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self._target_encoder = None
        self._train_idx = []
        self._val_idx = []
        self._tes = []
        self._test_te = TargetEncoder(
            columns=self.columns,
            noise_level=noise_level,
            min_samples_leaf=min_samples_leaf,
            smoothing=smoothing,
        )

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        self._test_te.fit(x, y)
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        if y is None:
            return self._test_te.transform(x)

        # Compute means for each folder
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        for train_idx, val_idx in kf.split(x):
            self._train_idx.append(train_idx)
            self._val_idx.append(val_idx)

            te = TargetEncoder(
                columns=self.columns,
                noise_level=self.noise_level,
                min_samples_leaf=self.min_samples_leaf,
                smoothing=self.smoothing,
            )
            self._tes.append(te.fit(x.iloc[train_idx, :], y=y.iloc[train_idx]))

        # Apply means across folds
        dfs = []

        for i, val_idx in enumerate(self._val_idx):
            target_encoder = self._tes[i]
            df_val_idx = x.iloc[val_idx, :]
            dfs.append(target_encoder.transform(df_val_idx, y=None))
        return pd.concat(dfs)

    def fit_transform(self, x: pd.DataFrame, y=None, **fit_params):
        return self.fit(x, y).transform(x, y)


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Encode as 0 or 1 categorical columns with 2 cardinalities
    """

    def __init__(self, columns: List[str] = None, strs_to_check: List[str] = None):
        self.columns = BINARY_COLUMNS if columns is None else columns
        self.strs_to_check = STRS_TO_CHECK if strs_to_check is None else strs_to_check

    def fit(self, x: pd.DataFrame = None, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        df_copy = x.copy()
        for col, s in zip(self.columns, self.strs_to_check):
            df_copy[col + "_feature"] = df_copy[col].apply(
                lambda val: 1 if s in val else 0
            )
        return df_copy

    def fit_transform(self, x: pd.DataFrame, y=None, **fit_params):
        return self.fit(x, y).transform(x, y)


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encode low categorical columns with one hot encoding
    """

    def __init__(self, columns: List[str] = None):
        self.columns = ONE_HOT_COLUMNS if columns is None else columns
        self._columns_maps = {}

    def _map(self, s: str, column: str) -> tuple:
        return tuple([int(s == val) for val in self._columns_maps[column]])

    def fit(self, x: pd.DataFrame = None, y: pd.Series = None):
        for col in self.columns:
            uniques = x[col].unique()
            self._columns_maps[col] = uniques

        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        x_copy = x.copy()
        for col in self.columns:
            f_cols = [
                col + str(i) + "_feature" for i in range(len(self._columns_maps[col]))
            ]
            x_copy[f_cols] = pd.DataFrame(
                x_copy[col].apply(lambda s: self._map(s, col)).tolist()
            )

        return x_copy

    def fit_transform(self, x: pd.DataFrame, y=None, **fit_params):
        return self.fit(x, y).transform(x, y)
