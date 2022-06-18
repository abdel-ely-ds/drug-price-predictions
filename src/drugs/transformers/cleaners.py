import re
from typing import List

import pandas as pd
import unidecode
from sklearn.base import BaseEstimator, TransformerMixin

from drugs.utils.constants import DATES_COLUMNS, MARKETING_YEAR, TEXT_COLUMNS


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
        s_1 = s.lower()
        s_1 = re.sub(",", "", s_1)
        s_1 = re.sub(r"[()]", "", s_1)
        return unidecode.unidecode(s_1)

    def fit(self, df, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col in self.columns:
            df_copy[col] = df_copy[col].apply(self.normalize_text)
        return df_copy


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

    def fit(self, df, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col in self.columns:
            df_copy[col] = df_copy[col].apply(self.to_datetime)

        # df_copy[MARKETING_YEAR] = df_copy[self.columns[0]].dt.year
        return df_copy


class DropColumnsCleaner(BaseEstimator, TransformerMixin):
    """
    Drop columns to keep only features
    """

    def __init__(self):
        self.columns = None

    def fit(self, df, y=None):
        self.columns = [col for col in df.columns if "feature" not in col]
        return self

    def transform(self, df: pd.DataFrame):
        return df.drop(columns=self.columns)
