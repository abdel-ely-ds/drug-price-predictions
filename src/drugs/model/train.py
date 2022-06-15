import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from drugs.constants import DRUG_ID, PRICE
from drugs.transformers.cleaners import DateCleaner, DropColumnsCleaner, TextCleaner
from drugs.transformers.encoders import BinaryEncoder, HighCardEncoder
from drugs.transformers.extractors import DescriptionExtractor


# Todo merge them
def _merge_dfs(raw_df: pd.DataFrame, ingredient_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([raw_df, ingredient_df])


class Trainer:
    run_id: int = 1

    def __init__(
        self,
        model,
        output_dir: str,
    ):
        self.run_id += 1
        self.model = model
        self.output_dir = output_dir
        self._processing_pipe = self._make_processing_pipeline()

    @property
    def processing_pipe(self) -> Pipeline:
        return self._processing_pipe

    @staticmethod
    def _make_processing_pipeline() -> Pipeline:
        pipe = Pipeline(
            [
                ("text_cleaner", TextCleaner()),
                ("date_cleaner", DateCleaner()),
                ("description_extractor", DescriptionExtractor),
                ("high_card_encoder", HighCardEncoder()),
                ("binary_encoder", BinaryEncoder()),
                ("drop_columns_cleaner", DropColumnsCleaner),
            ]
        )
        return pipe

    def train(
        self, raw_df: pd.DataFrame, ingredient_df: pd.DataFrame, verbose: bool = True
    ) -> None:
        final_df = _merge_dfs(raw_df, ingredient_df)

        train, val = train_test_split(final_df, test_size=0.2, random_state=2022)
        y_train, y_val = train[PRICE], val[PRICE]

        self._processing_pipe.fit(train)
        x_train = self._processing_pipe.transform(train)
        x_val = self._processing_pipe.transform(val)

        self.model.fit(x_train, y_train)

        if verbose:
            print("=" * 100)
            print(f"model scored on train: {self.model.score(train, x_train, y_train)}")
            print(f"model scored on val: {self.model.score(train, x_val, y_val)}")
            print("=" * 100)

    def predict(
        self, raw_df: pd.DataFrame, ingredient_df: pd.DataFrame
    ) -> pd.DataFrame:
        final_df = _merge_dfs(raw_df, ingredient_df)
        x = self._processing_pipe.transform(final_df)
        final_df["price"] = self.model.predict(x)
        return final_df[[DRUG_ID, PRICE]]
