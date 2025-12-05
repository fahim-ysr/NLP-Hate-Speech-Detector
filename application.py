# Importing required modules

import sys
from exception import CustomException
from constants import *
from pipelines.train import Training
from pipelines.prediction import Prediction

from fastapi import FastAPI
import uvicorn
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response


# Initializing FastAPI app
app = FastAPI()


@app.get("/", tags= ["authentication"])
async def index():
    """
    Interactive homepage for viewing all the options
    """
    return RedirectResponse(url= "/docs")


@app.get("/train")
async def train():
    """
    Executes the training pipeline. Downloads data from Google Cloud Storage -> Cleans it -> Trains the LSTM model
    """
    try:
        train_pipeline = Training()
        train_pipeline.execute_pipeline()
        return Response("Training successful!")


    # Exception handling
    except Exception as e:
        raise Response(f"Error Occured! {e}")
    

@app.post("/predict")
async def predict(user_input):
    """
    Takes user input (text and/or emoji) and returns "Hate Speech" or "No Hate". Downloads best model from Google Cloud Storage -> Cleans the user input -> Runs it through LSTM model -> Returns the preddiction
    """
    try:
        prediction = Prediction()
        user_input = prediction.execute_pipeline(user_input)
        return user_input


    # Exception handling
    except Exception as e:
        raise CustomException(e, sys) from e
    

# Runs when executed as a script
if __name__ == "__main__":
    uvicorn.run(app, host= HOST, port= PORT)