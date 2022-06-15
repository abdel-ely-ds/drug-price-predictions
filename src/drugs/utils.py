from __future__ import annotations

import pandas as pd


# Todo merge them
def merge_dfs(raw_df: pd.DataFrame, ingredient_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([raw_df, ingredient_df])


def get_latest_run_id(output_dir: str) -> int | None:
    pass
