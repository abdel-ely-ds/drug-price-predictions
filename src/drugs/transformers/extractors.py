import re
from collections import OrderedDict
from typing import List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from drugs.utils.constants import ALL_LABELS, DESCRIPTION_COLUMN


class DescriptionExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extraction from description
    """

    def __init__(self, labels: List[str] = None):
        self._labels = ALL_LABELS if labels is None else labels
        self._pattern = "\d+\.*\d* [a-z]+"

    @property
    def labels(self) -> List[str]:
        return self._labels

    def match(self, description: str) -> Tuple[int]:
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

    def fit(self):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        (
            df_copy[self._labels[0]],
            df_copy[self._labels[1]],
            df_copy[self._labels[2]],
            df_copy[self._labels[3]],
            df_copy[self._labels[4]],
            df_copy[self._labels[5]],
            df_copy[self._labels[6]],
            df_copy[self._labels[7]],
            df_copy[self._labels[8]],
            df_copy[self._labels[9]],
            df_copy[self._labels[10]],
            df_copy[self._labels[11]],
            df_copy[self._labels[12]],
            df_copy[self._labels[13]],
            df_copy[self._labels[14]],
        ) = zip(*df_copy[DESCRIPTION_COLUMN].map(self.match))
        return df_copy
