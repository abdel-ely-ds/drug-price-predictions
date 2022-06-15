from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HighCardEncoder(BaseEstimator, TransformerMixin):
    """
    Encode high cardinality transformers with a mean/median/quantile value of the price
    """

    def __init__(self, columns: List[str]):
        self.columns = columns
        self.columns_dict = {}

    def fit(self, df: pd.DataFrame, agg_func: str = "mean"):
        for col in self.columns:
            col_map = df.groupby(col).price.agg(func=agg_func)
            self.columns_dict[col] = col_map
        return self

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col in self.columns:
            df_copy[col + "_feature"] = df_copy[col].map(self.columns_dict[col])
            df_copy.drop(columns=[col], inplace=True)
        return df_copy


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Encode as 0 or 1 categorical columns with 2 cardinalities
    """

    def __init__(self, columns: List[str], strs_to_check: List[str]):
        self.columns = columns
        self.strs_to_check = strs_to_check

    def fit(self):
        return self

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col, s in zip(self.columns, self.strs_to_check):
            df_copy[col + "_feature"] = df_copy[col].apply(lambda x: 1 if s in x else 0)
            df_copy.drop(columns=[col], inplace=True)
        return df_copy
