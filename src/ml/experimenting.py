from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from prefect import flow, task
from loguru import logger
import mlflow
import pandas as pd
import os
import pickle

from src.etl.helper import load_config, dump_pickle


# Define mlflow parameters for tracking
#mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_tracking_uri("http://127.0.0.1:5000")


@task(name="Read data")
def read_data(data_path):
    data=pd.read_csv(data_path)

    return data


@task(retries=3, retry_delay_seconds=2, name="Splitting into train and validation sets")
def create_train_and_validation_sets(
    df: pd.DataFrame, model_data_path, target: str = "Spam", save=True, 
) -> pd.DataFrame:
    logger.info("create_train_and_validation_sets")

    X = df.Message
    y = df[target]

    # Split data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Save training and validation sets
    if save:
        logger.info(f"Saving training and validation datasets to path: {model_data_path}")

        # Saving files
        os.makedirs(model_data_path, exist_ok=True)

        dump_pickle((X_train, y_train), os.path.join(model_data_path, "train.pkl"))
        dump_pickle((X_val, y_val), os.path.join(model_data_path, "val.pkl"))

    return X_train, X_val, y_train, y_val


@task(name ="Calculate and store metrics ")
def run_metrics(y_val, y_pred):
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Store the metrics values in MLflow
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)


@task(name ="Log model")
def log_model(model, model_name):
    mlflow.sklearn.log_model(model, model_name)


@flow(name="Experimenting Tracking Flow")
def run_model_experimenting(config) -> None:

    EXPERIMENT_NAME = config.model.experiment_name

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Read the training data to perform parameter tuning
    train_df = read_data(config.data.preprocessed + "/train_df.csv")

    # Create training and validation datasets for the hyperparameter tuning

    model_data_path = config.data.model
    X_train, X_val, y_train, y_val = create_train_and_validation_sets(train_df, model_data_path).result()

    estimator = DecisionTreeClassifier(max_depth=1)

    model_pool = {1: MultinomialNB(), 2:RandomForestClassifier(),3:AdaBoostClassifier(estimator=estimator, n_estimators=50)}


    #mlflow.sklearn.autolog()
   
    for i, model in model_pool.items():

        with mlflow.start_run():
            logger.info(f"Run {i} experiment")

            model_pipeline=Pipeline([('vectorizer',CountVectorizer()),('nb',model)])

            model_pipeline.fit(X_train,y_train)

            y_pred = model_pipeline.predict(X_val)

            run_metrics(y_val, y_pred)
            mlflow.log_param("model_name", f"{model}" )
            mlflow.sklearn.log_model(model_pipeline, artifact_path=f"model_{i}")

            print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
            #log_model(model_pipeline, model_name)
            #mlflow.log_artifact("model", model)

if __name__ == "__main__":
    config = load_config()
    run_model_experimenting(config)


