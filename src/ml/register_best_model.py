# Import python libraries
import os
import sys

import mlflow
from loguru import logger
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
import yaml

CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)


from src.etl.helper import load_config


@task(name="Update config with best model run id")
def update_config(run_id):
    import yaml

    with open("./config/main.yaml") as f:
        config = yaml.safe_load(f)

        config["model"]["best_model_run_id"] = run_id

    with open("./config/main.yaml", "w") as f:
        yaml.dump(config, f)


@task(name="Copy model for deployment")
def prepare_deployment(run_id):
    model_path = f"./mlartifacts/1/{run_id}/artifacts/model/"

    import shutil

    source_path = model_path + "/model.pkl"
    destination_path = "./deployment/"

    # Use shutil.copy() to copy the file
    shutil.copy(source_path, destination_path)


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
    logger.info(
        f"Getting the last active mlflow experiments from experiment: {EXPERIMENT_NAME}"
    )

    client = MlflowClient()

    # Select the model with the lowest test RMSE (i.e. best model)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=["metrics.test_rmse DESC"],
    )[0]

    # Register the best model in model registry of MLflow
    best_model_uri = "runs:/{}/model".format(best_run.info.run_id)
    mlflow.register_model(
        model_uri=best_model_uri, name="best-spam-classification-model"
    )

    print(best_model_uri)

    update_config(best_run.info.run_id)

    prepare_deployment(best_run.info.run_id)

    logger.info("Finishing registering the best model procedure")


if __name__ == "__main__":
    config = load_config()
    register_best_model(config)
