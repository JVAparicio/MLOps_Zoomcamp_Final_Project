
import os
import sys

import pandas as pd


def test_encode_target():
    # Create a sample DataFrame
    data = [["ham", "Not a spam email"],
            ["spam", "A spam email"]]

    columns = ["Category", "Message"]

    df = pd.DataFrame(data, columns=columns)

    print(df)


    # Call the function under test
    df['Spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)

    # Assert that the returned value is a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Assert that the enconded values are correct
    assert df["Spam"][0] == 0
    assert df["Spam"][1] == 1
