import re
from collections import OrderedDict
from typing import List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from drugs.constants import DESCRIPTION_COLUMN


class DescriptionExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extraction from description
    """

    def __init__(self):
        self._labels = [
            "plaquette",
            "stylo",
            "tube",
            "seringue",
            "cachet",
            "gelule",
            "flacon",
            "ampoule",
            "ml",
            "g",
            "pilulier",
            "comprime",
            "film",
            "poche",
            "capsule",
        ]
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
            df_copy["plaquette"],
            df_copy["stylo"],
            df_copy["tube"],
            df_copy["tube"],
            df_copy["seringue"],
            df_copy["cachet"],
            df_copy["gelule"],
            df_copy["flacon"],
            df_copy["ampoule"],
            df_copy["ml"],
            df_copy["g"],
            df_copy["pilulier"],
            df_copy["comprime"],
            df_copy["film"],
            df_copy["poche"],
            df_copy["capsule"],
        ) = zip(*df_copy[DESCRIPTION_COLUMN].map(self.match))
        return df_copy
