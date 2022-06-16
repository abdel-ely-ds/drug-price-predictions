from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from drugs.constants import (
    ACTIVE_INGREDIENT,
    DRUG_ID,
    HIGH_CARD_COLUMNS,
    ONE_HOT_COLUMNS,
    PRICE,
    STRS_TO_CHECK,
)


class IngredientsEncoder(BaseEstimator, TransformerMixin):
    """
    Encode ingredients with a mean/median/quantile value of the price
    """

    def __init__(self, column: str = ACTIVE_INGREDIENT):
        self.column = column
        self.ingredient_price_map = {}

    @staticmethod
    def _make_ingredient_prices(x: pd.Series, top_k: int = 5) -> List[float]:
        if len(x) >= top_k:
            return x.iloc[0:top_k].tolist()
        else:
            x_list = x.tolist()
            return x_list + [0 for _ in range(top_k - len(x_list))]

    def fit(
        self, df: pd.DataFrame, df_ingredient: pd.DataFrame, agg_func: str = "mean"
    ):
        df_ingredient_copy = df_ingredient.copy()
        df_ingredient_copy[PRICE] = df_ingredient_copy.drug_id.map(
            df.set_index(DRUG_ID)[PRICE]
        )
        df_ingredient_copy["nb_ingredients"] = df_ingredient_copy.groupby(DRUG_ID)[
            self.column
        ].transform("nunique")
        df_ingredient_copy["ingredient_price"] = (
            df_ingredient_copy[PRICE] / df_ingredient_copy.nb_ingredients
        )
        self.ingredient_price_map = (
            df_ingredient_copy.groupby(self.column)
            .ingredient_price.agg(func=agg_func)
            .to_dict()
        )
        return self

    def transform(
        self, df: pd.DataFrame, df_ingredients: pd.DataFrame, top_k: int = 5
    ) -> pd.DataFrame:

        f_cols = [f"ingredient_{i}_feature" for i in range(top_k)]
        df_ingredients_copy = df_ingredients.copy()
        df_ingredients_copy["ingredient_price"] = df_ingredients_copy[self.column].map(
            self.ingredient_price_map
        )
        features_df = (
            df_ingredients_copy.sort_values("ingredient_price", ascending=True)
            .groupby(DRUG_ID, sort=False)
            .agg({"ingredient_price": lambda x: self._make_ingredient_prices(x, top_k)})
            .reset_index()
        )
        features_df[f_cols] = pd.DataFrame(features_df.ing_feature.tolist())
        return df.merge(features_df.drop("ingredient_price"), axis=1)


class HighCardEncoder(BaseEstimator, TransformerMixin):
    """
    Encode high cardinality transformers with a mean/median/quantile value of the price
    """

    def __init__(self, columns: List[str] = None):
        self.columns = HIGH_CARD_COLUMNS if columns is None else columns
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
        return df_copy


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Encode as 0 or 1 categorical columns with 2 cardinalities
    """

    def __init__(self, columns: List[str] = None, strs_to_check: List[str] = None):
        self.columns = ONE_HOT_COLUMNS if columns is None else columns
        self.strs_to_check = STRS_TO_CHECK if strs_to_check is None else strs_to_check

    def fit(self):
        return self

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col, s in zip(self.columns, self.strs_to_check):
            df_copy[col + "_feature"] = df_copy[col].apply(lambda x: 1 if s in x else 0)
        return df_copy
