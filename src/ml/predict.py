import mlflow
from loguru import logger
from fastapi import FastAPI
import numpy as np


#mlflow.set_tracking_uri("http://127.0.0.1:5000")


RUN_ID = "e490ff72fbd1497e824a714ee545315f"


logged_model = f"./mlartifacts/1/{RUN_ID}/artifacts/model_3/"
## Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


#print(loaded_model.predict([emails]))


app = FastAPI()


#def predict(features):
#    preds = loaded_model.predict(features)
#    return preds



@app.get("/predict")
async def predict(email: str):
    pred = loaded_model.predict([email])

    if pred == [0]:
        prediction = 'Not Spam'
    else:
        prediction = 'Spam'

     
    # have to convert to int because numpy is not supported by fastapi : https://github.com/tiangolo/fastapi/issues/2293
    return {'prediction': prediction, 'model_version': RUN_ID}


if __name__ == "__main__":
    # Use this for debugging perpuses

    logger.debug("Running in developement mode. Do not run like this in production")
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")



