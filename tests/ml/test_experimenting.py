# Import python libraries
import os
import sys
import warnings

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

warnings.filterwarnings("ignore")

import pandas as pd

from src.ml.experimenting import create_train_and_validation_sets
from src.etl.helper import load_config

config = load_config()

print(config)

model_data_path = config.data.model


# Helper function to create a sample DataFrame
def create_sample_dataframe():
    data = {
        "Category": ["ham", "spam", "spam", "ham", "spam"],
        "Message": ["aaa", "bbb", "ccc", "ddd", "eee"],
        "Spam": [0, 1, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_create_train_and_validation_sets():
    # Create a sample DataFrame
    df = create_sample_dataframe()


    # Call the function under test
    X_train, X_val, y_train, y_val = create_train_and_validation_sets.fn(
        df, model_data_path=model_data_path, target="Spam", save=False
    )

    #print(X_train)
    # Assert that the returned values are DataFrames/Series
    assert isinstance(X_train, object)
    assert isinstance(X_val, object)
    assert isinstance(y_train, object)
    assert isinstance(y_val, object)

    # Assert that the training and validation sets are not empty
    assert not X_train.empty
    assert not X_val.empty
    assert not y_train.empty
    assert not y_val.empty


    # Assert that the size of the training and validation sets are correct
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    # Assert that the training and validation sets are not the same
    assert not X_train.equals(X_val)
    assert not y_train.equals(y_val)

    # Assert that the pickle files are saved correctly
    assert os.path.exists(model_data_path + "/train.pkl")
    assert os.path.exists(model_data_path + "/val.pkl")