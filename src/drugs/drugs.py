import logging
import os
from typing import List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from drugs.transformers.cleaners import (
    DateCleaner,
    DropColumnsCleaner,
    DropDuplicatesCleaner,
    TextCleaner,
)
from drugs.transformers.encoders import (
    BinaryEncoder,
    DescriptionEncoder,
    OneHotEncoder,
    PercentageEncoder,
    TargetEncoderCV,
)
from drugs.utils.constants import (
    DRUG_ID,
    MODEL_DIRECTORY,
    MODEL_NAME,
    PIPELINE_DIRECTORY,
    PIPELINE_NAME,
    PREDICTION_DIRECTORY,
    PREDICTION_NAME,
    PRICE,
    SEED,
)


class Drugs:
    """
    Class responsible for training and inference
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        model: RandomForestRegressor = None,
        processing_pipeline: Pipeline = None,
        selected_features: List[str] = None,
    ):
        self._model = self._make_model() if model is None else model
        self._processing_pipe = (
            self._make_processing_pipeline()
            if processing_pipeline is None
            else processing_pipeline
        )

    @property
    def processing_pipe(self) -> Pipeline:
        return self._processing_pipe

    @property
    def model(self) -> RandomForestRegressor:
        return self._model

    @staticmethod
    def _make_model() -> RandomForestRegressor:
        return RandomForestRegressor(
            max_depth=8,
            min_samples_leaf=4,
            min_samples_split=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
            n_estimators=110,
        )

    @staticmethod
    def _make_processing_pipeline() -> Pipeline:
        pipe = Pipeline(
            [
                ("text_cleaner", TextCleaner()),
                ("date_cleaner", DateCleaner()),
                ("percentage_encoder", PercentageEncoder()),
                ("one_hot_encoder", OneHotEncoder()),
                ("binary_encoder", BinaryEncoder()),
                ("target_encoder", TargetEncoderCV()),
                ("description_encoder", DescriptionEncoder()),
                ("drop_duplicates", DropDuplicatesCleaner()),
                ("drop_columns", DropColumnsCleaner()),
            ]
        )
        return pipe

    def fit(
        self,
        df: pd.DataFrame,
        df_ingredient: pd.DataFrame,
    ) -> None:

        y_train = df[PRICE]

        train = df.merge(df_ingredient)

        x_train = self._processing_pipe.fit_transform(train, train[PRICE])
        self._model.fit(x_train, y_train)

        self.logger.info("training finished!")

    def predict(self, df: pd.DataFrame, df_ingredient: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.merge(df_ingredient)
        x = self._processing_pipe.transform(df_copy)
        prices = self._model.predict(x)
        return pd.DataFrame.from_dict({DRUG_ID: df[DRUG_ID], PRICE: prices})

    def save_artifacts(self, output_dir: str) -> None:
        joblib.dump(
            self._processing_pipe,
            os.path.join(output_dir, PIPELINE_DIRECTORY, PIPELINE_NAME),
        )
        joblib.dump(
            self._model,
            os.path.join(output_dir, MODEL_DIRECTORY, MODEL_NAME),
        )
        self.logger.info(f"artifacts saved successfully to {output_dir}")

    @staticmethod
    def save_predictions(predictions: pd.DataFrame, output_dir: str) -> None:
        predictions.to_csv(
            os.path.join(output_dir, PREDICTION_DIRECTORY, PREDICTION_NAME + ".csv"),
            index=False,
        )

    def load_artifacts(
        self,
        from_dir: str,
    ) -> None:
        self._model = joblib.load(os.path.join(from_dir, MODEL_DIRECTORY, MODEL_NAME))
        self._processing_pipe = joblib.load(
            os.path.join(from_dir, PIPELINE_DIRECTORY, PIPELINE_NAME)
        )
