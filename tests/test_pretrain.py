import pandas as pd
import pytest

from animated_octo_guide.train import train_test_val_split


class TestPreTrain:
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

    def test_data_shape(self, preprocess_data):
        """Test to check if the number of training features are correct"""
        x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data
        assert x_train.shape[1] == 13
        assert x_val.shape[1] == 13
        assert x_test.shape[1] == 13
