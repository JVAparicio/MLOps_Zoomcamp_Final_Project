# Import python libraries
import os
import sys
import warnings
import pandas as pd
import pickle
import mlflow

from loguru import logger

from helper import load_config

warnings.filterwarnings("ignore")

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report

# Define entry point for paths
CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)

# Import helper functions


def run_monitoring(config) -> None:
    """Run the data drift and performance monitor process.

    Args:
        config_path (str): Configuration

    Returns:
        None
    """
    logger.info("Starting Monitoring procedure.")


    # Unpack config file
    reference_data = config.data.preprocessed
    current_data = config.data.current
    #model_run_path = config.inference.model_run_path
    #categorical_features = config["preprocessing"]["cat_variables"]


    #2. Read reference and current data
    reference_df = pd.read_csv(reference_data + "/train_df.csv")
    current_df = pd.read_csv(current_data + "/monitor_df.csv")


    # 3. Load model
    RUN_ID = config.model.best_model_run_id

    model = f"./mlartifacts/1/{RUN_ID}/artifacts/model_3/"
    ## Load model as a PyFuncModel.
    model = mlflow.pyfunc.load_model(model)

    # 4. Predict on current and reference data
    reference_pred = model.predict(reference_df["Message"])
    current_pred = model.predict(current_df["Message"])
    

    current_df["prediction"] = current_pred
    reference_df["prediction"] = reference_pred

    # Set evidently column mapping
    column_mapping = ColumnMapping()

    column_mapping.target = "spam"
    column_mapping.prediction = "prediction"
    #column_mapping.categorical_features = categorical_features

    ## Create regression report of model performance
    #logger.info("Get the regression report.")
    #regression_report = Report(metrics=[RegressionPreset()])
    #regression_report.run(
    #    reference_data=reference_df,
    #    current_data=current_df,
    #    column_mapping=column_mapping,
    #)

    # Create data drift report
    logger.info("Get the drift report.")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )
    logger.info("Saving reports...")

    report_path = config.monitoring.report
    #regression_report.save_html("./workspace/test_suite.html")
    drift_report.save_html(report_path + "/test_drift_report.html")

    logger.info("Monitoring procedure finished.")



if __name__ == "__main__":
    config = load_config()
    run_monitoring(config)