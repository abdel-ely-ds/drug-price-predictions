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

    run_id: int = 1
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        model=None,
        processing_pipeline: Pipeline = None,
    ):
        self.model = XGBRegressor(random_state=SEED) if model is None else model
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
        self.run_id += 1

        y_train = df[PRICE]
        train = df.merge(df_ingredient)

        self._processing_pipe.fit(train)
        x_train = self._processing_pipe.transform(train)

        if val_df is not None and val_df_ingredient is not None:
            y_val = val_df[PRICE]
            val = val_df.merge(val_df_ingredient)
            x_val = self._processing_pipe.transform(val)
            self.model.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )

        else:
            self.model.fit(x_train, y_train)

        self.logger.info("training finished!")

    # ToDo fix this
    def predict(self, df: pd.DataFrame, df_ingredient: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.merge(df_ingredient)
        x = self._processing_pipe.transform(df_copy)
        ret = df.copy()[DRUG_ID]
        return self.model.predict(x)

    def plot_learning_curve(self):
        results = self.model.evals_result()
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
            os.path.join(
                output_dir, PIPELINE_DIRECTORY, PIPELINE_NAME + "_" + str(self.run_id)
            ),
        )
        joblib.dump(
            self.model,
            os.path.join(
                output_dir, MODEL_DIRECTORY, MODEL_NAME + "_" + str(self.run_id)
            ),
        )
        self.logger.info(f"artifacts saved successfully to {output_dir}")

    def save_predictions(self, predictions: pd.DataFrame, output_dir: str) -> None:
        joblib.dump(
            predictions,
            os.path.join(
                output_dir,
                PREDICTION_DIRECTORY,
                PREDICTION_NAME + "_" + str(self.run_id),
            ),
        )

    def load_artifacts(
        self,
        from_dir: str,
        run_id: int,
    ) -> None:
        self.model = joblib.load(
            os.path.join(from_dir, MODEL_DIRECTORY, MODEL_NAME + "_" + str(run_id))
        )
        self._processing_pipe = joblib.load(
            os.path.join(
                from_dir, PIPELINE_DIRECTORY, PIPELINE_NAME + "_" + str(run_id)
            )
        )
