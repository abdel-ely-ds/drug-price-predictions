import re
from typing import List

import pandas as pd
import unidecode
from sklearn.base import BaseEstimator, TransformerMixin

from drugs.utils.constants import (
    DATES_COLUMNS,
    DRUG_ID,
    HIGH_CARD_COLUMNS,
    PHARMACY_COLUMN,
    SELECTED_FEATURES,
    TEXT_COLUMNS,
    YEAR,
)


class PharmacyCleaner(BaseEstimator, TransformerMixin):
    """Clean pharmacy column by taking only the first pharmacy"""

    def __init__(self, column: str = PHARMACY_COLUMN):
        self.column = column

    @staticmethod
    def normalize_pharmacy(s: str) -> str:
        """
        clean pharmacy column
        :param s: string to clean
        :return: cleaned version of text
        """
        s_1 = s.lower().strip()
        return s_1.split(",")[0]

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        x_copy = x.copy()
        x_copy[self.column] = x_copy[self.column].apply(self.normalize_pharmacy)
        return x_copy

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None, **fit_params):
        return self.fit(x, y).transform(x, y)


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Clean description
    """

    def __init__(self, columns: List[str] = None):
        self.columns = TEXT_COLUMNS if columns is None else columns

    @staticmethod
    def normalize_text(s: str) -> str:
        """
        Clean and standardize text
        :param s: string to clean
        :return: cleaned version of text
        """
        s_1 = s.lower().strip()
        s_1 = re.sub(",", "", s_1)
        s_1 = re.sub(r"[()]", "", s_1)
        return unidecode.unidecode(s_1)

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        x_copy = x.copy()
        for col in self.columns:
            x_copy[col] = x_copy[col].apply(self.normalize_text)
        return x_copy

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None, **fit_params):
        return self.fit(x, y).transform(x, y)


class DateCleaner(BaseEstimator, TransformerMixin):
    """
    Put date columns on the right format as adds a year column
    """

    def __init__(self, columns: List[str] = None):
        self.columns = DATES_COLUMNS if columns is None else columns

    @staticmethod
    def to_datetime(series: pd.Series, date_format: str = "%Y%m%d") -> pd.Series:
        """
        :param series: a series dataframe of dates that written in format <format>
        :param date_format: format of the date in the series
        :return: a series of datetime
        """
        return pd.to_datetime(series, format=date_format)

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        x_copy = x.copy()
        for col in self.columns:
            x_copy[col] = x_copy[col].apply(self.to_datetime)

        x_copy[YEAR] = x_copy[self.columns[0]].dt.year
        return x_copy

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None, **fit_params):
        return self.fit(x, y).transform(x, y)


class DropColumnsCleaner(BaseEstimator, TransformerMixin):
    """
    Drop columns to keep only features
    """

    def __init__(self):
        self.columns = SELECTED_FEATURES
        self.drop_columns = None

    def fit(self, x, y: pd.Series = None):
        self.drop_columns = [col for col in x.columns if col not in self.columns]
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        return x.drop(columns=self.drop_columns)

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None, **fit_params):
        return self.fit(x, y).transform(x, y)


class DropDuplicatesCleaner(BaseEstimator, TransformerMixin):
    """
    Drop columns to keep only features
    """

    def __init__(self, columns: List[str] = None):
        self.columns = HIGH_CARD_COLUMNS if columns is None else columns

    def fit(self, x, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        x_1 = x.groupby(DRUG_ID, sort=False)[self.columns].mean().reset_index()
        x_2 = x.drop(columns=self.columns).drop_duplicates(subset=DRUG_ID)
        return x_1.merge(x_2)

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None, **fit_params):
        return self.fit(x, y).transform(x, y)
