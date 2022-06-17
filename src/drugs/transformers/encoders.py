from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from drugs.utils.constants import (
    ACTIVE_INGREDIENT,
    DRUG_ID,
    HIGH_CARD_COLUMNS,
    ONE_HOT_COLUMNS,
    PHARMACY,
    PRICE,
    REIMBURSEMENT_RATE,
    STRS_TO_CHECK,
)


class IngredientEncoder(BaseEstimator, TransformerMixin):
    """
    Encode ingredients with a mean/median/quantile value of the price

    -> To estimate how much an ingredient costs in a drug we take the price of the drug and divide
    it by the number of ingredients it contains. Then that ingredient would be assigned a feature which
    is the mean price over all the dataset.

    -> Therefore the ingredients of a drug would be represented as a sorted list of length <top_k>
    """

    def __init__(self, column: str = ACTIVE_INGREDIENT):
        self.column = column
        self.ingredient_price_map = {}

    @staticmethod
    def _make_ingredient_prices(x: pd.Series, top_k: int = 3) -> List[float]:
        if len(x) >= top_k:
            return x.iloc[0:top_k].tolist()
        else:
            x_list = x.tolist()
            return x_list + [0 for _ in range(top_k - len(x_list))]

    def fit(
        self,
        df: pd.DataFrame,
        y=None,
    ):
        df_copy = df.copy()
        df_copy["nb_ingredients"] = df_copy.groupby(DRUG_ID, sort=False)[
            self.column
        ].transform("nunique")
        df_copy["ingredient_price"] = df_copy[PRICE] / df_copy.nb_ingredients
        self.ingredient_price_map = (
            df_copy.groupby(self.column, sort=False)
            .ingredient_price.agg(func="mean")
            .to_dict()
        )
        return self

    def transform(self, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:

        f_cols = [f"ingredient{i}_feature" for i in range(top_k)]
        df_copy = df.copy()
        df_copy["ingredient_price"] = df_copy[self.column].map(
            self.ingredient_price_map
        )
        features_df = (
            df_copy.sort_values("ingredient_price", ascending=True)
            .groupby(DRUG_ID, sort=False)
            .agg({"ingredient_price": lambda x: self._make_ingredient_prices(x, top_k)})
            .reset_index()
        )
        features_df[f_cols] = pd.DataFrame(features_df.ingredient_price.tolist())
        df_full = df_copy.merge(features_df.drop("ingredient_price", axis=1))
        return df_full.drop(ACTIVE_INGREDIENT, axis=1).drop_duplicates(subset=[DRUG_ID])


class PharmacyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode pharmacy companies with a mean value of the price
    """

    def __init__(self, column: str = PHARMACY):
        self.column = column
        self.ingredient_price_map = {}

    def fit(
        self,
        df: pd.DataFrame,
        y=None,
    ):
        return self

    def transform(self, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
        return


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Encode high cardinality transformers with a mean/median/quantile value of the price
    """

    def __init__(self, columns: List[str] = None):
        self.columns = HIGH_CARD_COLUMNS if columns is None else columns
        self.columns_dict = {}

    def fit(self, df: pd.DataFrame, y=None):
        for col in self.columns:
            col_map = df.groupby(col, sort=False)[PRICE].agg(func="mean")
            self.columns_dict[col] = col_map
        return self

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col in self.columns:
            df_copy[col + "_feature"] = df_copy[col].map(self.columns_dict[col])
        return df_copy


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Encode as 0 or 1 categorical columns with 2 cardinalities
    """

    def __init__(self, columns: List[str] = None, strs_to_check: List[str] = None):
        self.columns = ONE_HOT_COLUMNS if columns is None else columns
        self.strs_to_check = STRS_TO_CHECK if strs_to_check is None else strs_to_check

    def fit(self, df, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col, s in zip(self.columns, self.strs_to_check):
            df_copy[col + "_feature"] = df_copy[col].apply(lambda x: 1 if s in x else 0)
        return df_copy


class PercentageEncoder(BaseEstimator, TransformerMixin):
    """
    Encode percentages as float between 0 and 1
    """

    def __init__(self, column: str = REIMBURSEMENT_RATE):
        self.column = column

    def fit(self, df, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        df_copy[self.column + "_feature"] = (
            df_copy[self.column].str.replace("%", "").astype(int) / 100
        )

        return df_copy
