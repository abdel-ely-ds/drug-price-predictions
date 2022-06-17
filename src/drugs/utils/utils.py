from __future__ import annotations

import pandas as pd


# Todo merge them
def merge_dfs(raw_df: pd.DataFrame, ingredient_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([raw_df, ingredient_df])


def get_latest_run_id(output_dir: str) -> int | None:
    pass


class NoModeSpecified(Exception):
    """Exception raised for errors in no flag was specified
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="please specify one of the flags train or predict"):
        self.message = message
        super().__init__(self.message)


class MultipleModes(Exception):
    """Exception raised for errors in multiple flags were specified
    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message="please choose only one of flags train or predict not both at the same time",
    ):
        self.message = message
        super().__init__(self.message)
