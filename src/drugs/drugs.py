import logging
import os
from typing import List

import joblib
import matplotlib.pyplot as plt
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
    DescriptionEncoder,
    OneHotEncoder,
    PercentageEncoder,
    TargetEncoder,
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
    SELECTED_FEATURES,
)


class Drugs:
    """
    Class responsible for training and inference
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        model=None,
        processing_pipeline: Pipeline = None,
        selected_features: List[str] = None,
    ):
        self._model = self._make_model() if model is None else model
        self._processing_pipe = (
            self._make_processing_pipeline()
            if processing_pipeline is None
            else processing_pipeline
        )
        self.selected_features = (
            SELECTED_FEATURES if selected_features is None else selected_features
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
            max_depth=6,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
            n_estimators=500,
        )

    @staticmethod
    def _make_processing_pipeline() -> Pipeline:
        pipe = Pipeline(
            [
                ("text_cleaner", TextCleaner()),
                ("date_cleaner", DateCleaner()),
                ("percentage_encoder", PercentageEncoder()),
                ("one_hot_encoder", OneHotEncoder()),
                ("target_encoder", TargetEncoder()),
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
        self._model.fit(x_train[self.selected_features], y_train)

        self.logger.info("training finished!")

    def predict(self, df: pd.DataFrame, df_ingredient: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.merge(df_ingredient)
        x = self._processing_pipe.transform(df_copy)
        prices = self._model.predict(x[self.selected_features])
        return pd.DataFrame.from_dict({DRUG_ID: df[DRUG_ID], PRICE: prices})

    def plot_learning_curve(self):
        results = self._model.evals_result()
        epochs = len(results["validation_0"]["rmse"])
        x_axis = range(0, epochs)

        fig, ax = plt.subplots()
        ax.plot(x_axis, results["validation_0"]["rmse"], label="Train")
        ax.plot(x_axis, results["validation_1"]["rmse"], label="Val")
        ax.legend()
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.title("XGBoost learning curve")
        plt.show()

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
