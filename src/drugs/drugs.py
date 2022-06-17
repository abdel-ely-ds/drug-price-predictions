import logging
import os

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from drugs.transformers.cleaners import DateCleaner, DropColumnsCleaner, TextCleaner
from drugs.transformers.encoders import (
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
                early_stopping=20,
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

    @staticmethod
    def score(y_preds, y: pd.Series) -> float:
        """
        Computes rmse
        """
        return mean_squared_error(y_preds, y, squared=False)

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
