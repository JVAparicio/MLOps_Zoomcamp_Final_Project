import mlflow
from loguru import logger
from fastapi import FastAPI
import numpy as np
import os
import sys

CWD = os.getcwd()
os.chdir(CWD)
sys.path.append(CWD)


mlflow.set_tracking_uri("http://127.0.0.1:5000")

from src.etl.helper import load_config

# from helper import load_config

config = load_config()


RUN_ID = config.model.best_model_run_id


logged_model = f"./mlartifacts/1/{RUN_ID}/artifacts/model_3/"
## Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


app = FastAPI()


# def predict(features):
#    preds = loaded_model.predict(features)
#    return preds


@app.get("/predict")
async def predict(email: str):
    pred = loaded_model.predict([email])

    if pred == [0]:
        prediction = "Not Spam"
    else:
        prediction = "Spam"

    return {"prediction": prediction, "model_version": RUN_ID}


if __name__ == "__main__":
    logger.debug("Running in developement mode. Do not run like this in production")
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
