import pandas as pd
import pytest


class TestPetFinderData:
    @pytest.fixture
    def df(self):
        """Get the data from the url and load as pandas dataframe"""
        url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
        df = pd.read_csv(url)
        return df

    def test_columns_present(self, df):
        """Test to check if all the columns exists in the dataset"""
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
            "Adopted",
        ]
        assert set(cols) == set(df.columns)

    def test_null_values_in_row(self, df):
        """Test to verify that no row contains null value in the dataset"""
        assert df.isnull().values.any() == False

    def test_non_empty(self, df):
        """Test to check if the dataset has all the data samples"""
        assert len(df) == 11537
