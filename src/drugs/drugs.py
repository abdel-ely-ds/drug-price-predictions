import logging
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from drugs.transformers.cleaners import DateCleaner, DropColumnsCleaner, TextCleaner
from drugs.transformers.encoders import (
    DescriptionEncoder,
    IngredientEncoder,
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
    def model(self):
        return self._model

    @staticmethod
    def _make_model():
        return XGBRegressor(random_state=SEED)

    @staticmethod
    def _make_processing_pipeline() -> Pipeline:
        pipe = Pipeline(
            [
                ("text_cleaner", TextCleaner()),
                ("date_cleaner", DateCleaner()),
                ("percentage_encoder", PercentageEncoder()),
                ("target_encoder", TargetEncoder()),
                ("ingredient_encoder", IngredientEncoder()),
                ("description_encoder", DescriptionEncoder()),
                ("drop_columns", DropColumnsCleaner()),
            ]
        )
        return pipe

    def fit(
        self,
        df: pd.DataFrame,
        df_ingredient: pd.DataFrame,
        val_df: pd.DataFrame = None,
        val_df_ingredient: pd.DataFrame = None,
        verbose: bool = True,
        early_stopping_rounds: int = 20,
    ) -> None:

        y_train = df[PRICE]
        train = df.merge(df_ingredient)

        self._processing_pipe.fit(train)
        x_train = self._processing_pipe.transform(train)

        if val_df is not None and val_df_ingredient is not None:
            y_val = val_df[PRICE]
            val = val_df.merge(val_df_ingredient)
            x_val = self._processing_pipe.transform(val)
            self._model.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )

        else:
            self._model.fit(x_train, y_train)

        self.logger.info("training finished!")

    # ToDo fix this
    def predict(self, df: pd.DataFrame, df_ingredient: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.merge(df_ingredient)
        x = self._processing_pipe.transform(df_copy)
        prices = self._model.predict(x)

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
