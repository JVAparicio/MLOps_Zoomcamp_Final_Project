from loguru import logger
from fastapi import FastAPI
import numpy as np
import os
import sys
import pickle

with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

app = FastAPI()


@app.get("/predict")
async def predict(email: str):
    pred = loaded_model.predict([email])

    if pred == [0]:
        prediction = "Not Spam"
    else:
        prediction = "Spam"

    return {"prediction": prediction}


if __name__ == "__main__":
    logger.debug("Running in developement mode. Do not run like this in production")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
