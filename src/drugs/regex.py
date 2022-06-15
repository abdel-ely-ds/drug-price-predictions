import re
from typing import Dict


class DescriptionRegex:
    """
    Feature extraction from description
    """

    def __init__(self):
        self.labels = [
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
        self._quantity_pattern = "\d+\.*\d*"
        self._label_pattern = "[a-z]+"
        self._pattern = self._quantity_pattern + " " + self._label_pattern

    @property
    def pattern(self) -> str:
        return self._pattern

    def run(self, description: str) -> Dict[str, int]:
        """
        Find labels in description and their count
        """
        ret = {label: 0 for label in self.labels}
        matches = re.findall(self._pattern, description)
        for match in matches:
            quantity = re.search(self._quantity_pattern, match).group()
            label = re.search(self._label_pattern, match).group()
            ret[label] = quantity
        return ret
