import pandas as pd
import pytest
import xgboost as xgb

from animated_octo_guide.train import (get_dmatrix, load_model, predict,
                                       train_test_val_split)


class TestPostTrain:
    @pytest.fixture
    def df(self):
        """Get the data from the url and load as pandas dataframe"""
        url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
        df = pd.read_csv(url)
        return df

    @pytest.fixture
    def preprocess_data(self, df):
        """Get the data from the url and load as pandas dataframe"""
        # url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
        return train_test_val_split(df)

    @pytest.fixture
    def get_dmat_test(self, preprocess_data, df):
        """Get the classification matrices"""
        # url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
        x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data
        dtest_mat = xgb.DMatrix(x_test, enable_categorical=True)
        return dtest_mat

    @pytest.fixture
    def model(self):
        """XGBoost model"""
        model_path = "artifacts/model/xgboost_model.json"
        model = load_model(model_path)
        return model

    def test_output_shape(self, preprocess_data, get_dmat_test, model):
        x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data
        dtest_mat = get_dmat_test
        y_preds = model.predict(dtest_mat)
        assert x_test.shape[0] == y_preds.shape[0]

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                [
                    "Dog",
                    1,
                    "Mixed Breed",
                    "Male",
                    "Brown",
                    "No Color",
                    "Medium",
                    "Short",
                    "No",
                    "No",
                    "Healthy",
                    0,
                    1,
                ],
                "Yes",
            ),
            (
                [
                    "Dog",
                    1,
                    "Mixed Breed",
                    "Male",
                    "Black",
                    "Brown",
                    "Medium",
                    "Short",
                    "No",
                    "No",
                    "Healthy",
                    0,
                    7,
                ],
                "Yes",
            ),
        ],
    )
    def test_output(self, model, test_input, expected):
        """Test whether the outputs are correct or not for given input features"""
        cols = [
            "Type",
            "Age",
            "Breed1",
            "Gender",
            "Color1",
            "Color2",
            "MaturitySize",
            "FurLength",
            "Vaccinated",
            "Sterilized",
            "Health",
            "Fee",
            "PhotoAmt",
        ]
        df = pd.DataFrame(data=[test_input], columns=cols)
        dx_mat = get_dmatrix(df)
        result = predict(model, dx_mat)
        assert result[0] == expected

    def test_model_metadata(self, model):
        cols = [
            "Type",
            "Age",
            "Breed1",
            "Gender",
            "Color1",
            "Color2",
            "MaturitySize",
            "FurLength",
            "Vaccinated",
            "Sterilized",
            "Health",
            "Fee",
            "PhotoAmt",
        ]
        assert set(model.feature_names) == set(cols)
