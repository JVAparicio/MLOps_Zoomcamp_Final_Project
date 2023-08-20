# Import python libraries
import os
import sys


import mlflow
from loguru import logger
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


from helper import load_config, dump_pickle



# Define mlflow tracking parameters
mlflow.set_tracking_uri("http://127.0.0.1:5000")


@flow(name="Register Best ML Model")
def register_best_model(config) -> None:
    """Register the best model."

    Args:
        config

    Returns:
        None
    """
    EXPERIMENT_NAME = config.model.experiment_name

    # Get the all the runs from the hyperparameter experiment
    logger.info(f"Getting the last active mlflow experiments from experiment: {EXPERIMENT_NAME}")
   
    client = MlflowClient()

    # Select the model with the lowest test RMSE (i.e. best model)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=["metrics.test_rmse DESC"],
    )[0]

    # Register the best model in model registry of MLflow
    best_model_uri = "runs:/{}/model".format(best_run.info.run_id)
    mlflow.register_model(model_uri=best_model_uri, name="best-spam-classification-model")

    logger.info("Finishing registering the best model procedure")


if __name__ == "__main__":
    config = load_config()
    register_best_model(config)