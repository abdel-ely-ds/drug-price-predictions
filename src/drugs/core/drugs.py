import logging
import os

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from drugs.constants import (
    DRUG_ID,
    MODEL_DIRECTORY,
    MODEL_NAME,
    PIPELINE_DIRECTORY,
    PIPELINE_NAME,
    PREDICTION_DIRECTORY,
    PREDICTION_NAME,
    PRICE,
)
from drugs.core.transformers.cleaners import (
    DateCleaner,
    DropColumnsCleaner,
    TextCleaner,
)
from drugs.core.transformers.encoders import (
    BinaryEncoder,
    HighCardEncoder,
    PercentageEncoder,
)
from drugs.core.transformers.extractors import DescriptionExtractor


class Drugs:
    """
    Class responsible for training and inference
    """

    run_id: int = 1
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        model=None,
        processing_pipeline: Pipeline = None,
    ):
        self.model = (
            XGBRegressor(random_state=2022, eval_metric="mse")
            if model is None
            else model
        )
        self._processing_pipe = (
            self._make_processing_pipeline()
            if processing_pipeline is None
            else processing_pipeline
        )

    @property
    def processing_pipe(self) -> Pipeline:
        return self._processing_pipe

    @staticmethod
    def _make_processing_pipeline() -> Pipeline:
        pipe = Pipeline(
            [
                ("text_cleaner", TextCleaner()),
                ("date_cleaner", DateCleaner()),
                ("percentage_encoder", PercentageEncoder()),
                ("high_card_encoder", HighCardEncoder()),
                ("drop_columns", DropColumnsCleaner()),
            ]
        )
        return pipe

    def train(self, df: pd.DataFrame, verbose: bool = True) -> None:
        self.run_id += 1

        train, val = train_test_split(df, test_size=0.2, random_state=2022)
        y_train, y_val = train[PRICE], val[PRICE]

        self._processing_pipe.fit(train)
        x_train = self._processing_pipe.transform(train)
        x_val = self._processing_pipe.transform(val)

        self.model.fit(x_train, y_train)

        if verbose:
            print("=" * 100)
            print(f"model scored on train: {self.model.score(x_train, y_train)}")
            print(f"model scored on val: {self.model.score(x_val, y_val)}")
            print("=" * 100)

        self.logger.info("training finished!")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        x = self._processing_pipe.transform(df_copy)
        df_copy[PRICE] = self.model.predict(x)
        return df_copy[[DRUG_ID, PRICE]]

    def score(self, df: pd.DataFrame, y: pd.Series) -> float:
        """
        Computes rmse
        """
        return mean_squared_error(self.predict(df), y, squared=False)

    def save_artifacts(self, output_dir: str) -> None:
        joblib.dump(
            self._processing_pipe,
            os.path.join(
                output_dir, PIPELINE_DIRECTORY, PIPELINE_NAME, str(self.run_id)
            ),
        )
        joblib.dump(
            self.model,
            os.path.join(output_dir, MODEL_DIRECTORY, MODEL_NAME, str(self.run_id)),
        )
        self.logger.info(f"artifacts saved successfully to {output_dir}")

    def save_predictions(self, predictions: pd.DataFrame, output_dir: str) -> None:
        joblib.dump(
            predictions,
            os.path.join(
                output_dir, PREDICTION_DIRECTORY, PREDICTION_NAME, str(self.run_id)
            ),
        )

    def load_artifacts(
        self,
        from_dir: str,
        run_id: int,
    ) -> None:
        self.model = joblib.load(
            os.path.join(from_dir, MODEL_DIRECTORY, MODEL_NAME, str(run_id))
        )
        self._processing_pipe = joblib.load(
            os.path.join(from_dir, PIPELINE_DIRECTORY, PIPELINE_NAME, str(run_id))
        )
