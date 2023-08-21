# Import python libraries
import os
import sys

import pandas as pd


# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)


from src.etl.helper import load_config

from loguru import logger
from prefect import flow, task


@task(name="Read data")
def read_data(data_path):
    data = pd.read_csv(data_path)

    return data


@task(name="Encode Target")
def encode_target(data):
    data["Spam"] = data["Category"].apply(lambda x: 1 if x == "spam" else 0)

    return data


@task(name="Write preprocessed file")
def write_file(data) -> None:
    # Save the preprocessed training data
    save_path = os.path.join(config.data.preprocessed, "train_df.csv")

    logger.info(f"Saving preprocessed training data to {save_path}")

    os.makedirs(config.data.preprocessed, exist_ok=True)

    data.to_csv(save_path, index=False)


@flow(name="Running Preprocessing Flow")
def run_preprocessing(config) -> None:
    """Run the preprocessing pipeline step by step.

    Args:
        config (DictConfig): Config yaml

    Returns:
        None
    """
    logger.info("Running the preprocessing pipeline")

    print(config.data.raw)

    # Run preprocessing steps
    data = read_data(config.data.raw)
    data = encode_target(data)
    write_file(data)

    logger.info("Preprocessing pipeline completed")


if __name__ == "__main__":
    config = load_config()
    run_preprocessing(config)
