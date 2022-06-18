import mock
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from drugs.drugs import Drugs


@pytest.fixture()
def df():
    return pd.read_csv("../data/drugs_train.csv")


@pytest.fixture()
def df_ingredient():
    return pd.read_csv("../data/active_ingredients.csv")


@pytest.fixture()
def drugs():
    return Drugs()


class TestDrugs:
    def test__make_model(self, drugs):
        assert isinstance(drugs.model, XGBRegressor)

    def test__make_processing_pipeline(self, drugs):
        assert isinstance(drugs.processing_pipe, Pipeline)

    @mock.patch.object(Drugs, "_make_processing_pipeline")
    def test_fit_model(self, mocked_make_processing_pipeline, drugs, df, df_ingredient):
        mocked_make_processing_pipeline.return_value = mock.MagicMock()
        mocked_make_processing_pipeline.return_value.fit = None
        mocked_make_processing_pipeline.return_value.transform = None
        drugs.fit(df, df_ingredient)
        assert mocked_make_processing_pipeline.fit.assert_called_once()
