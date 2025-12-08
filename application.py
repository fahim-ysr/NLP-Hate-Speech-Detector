# Importing required modules

import sys
from exception import CustomException
from constants import *
from pipelines.train import Training
from pipelines.prediction import Prediction

from fastapi import FastAPI, Request
import uvicorn
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response, HTMLResponse, JSONResponse


# Create our FastAPI app
app = FastAPI(
    title="NLP-based Hate Speech Detector",
    description="NLP-based Hate Speech Detection Using LSTM Neural Networks",
    version="1.0"
)

# Set up Jinja2 templates (points to templates folder)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the main UI page where users can input text for analysis
    """

    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/docs-redirect", tags=["documentation"])
async def docs_redirect():
    """
    Redirects to the API documentation page
    """

    return RedirectResponse(url="/docs")


@app.post("/train")
async def train():
    """
    Executes the training pipeline. Downloads data from GCloud -> Cleans it -> Trains the LSTM model -> Evaluates it against the current best model -> Uploads if it is better
    """

    try:
        train_pipeline = Training()
        result = train_pipeline.execute_pipeline()
        
        if result is None:
            return JSONResponse(content={"status": "success", "message": "Training completed successfully!"})
        elif result.get("status") == "not_improved":
            return JSONResponse(content={"status": "warning", "message": result.get("message")})
        else:
            return JSONResponse(content={"status": "success", "message": result.get("message")})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.post("/predict")
async def predict(user_input):
    """
    Takes user text and returns whether it's hate speech or not
    """

    try:
        # Check if input is empty
        if not user_input or not user_input.strip():
            return JSONResponse(content={"status": "error", "message": "Please enter some text to analyze"}, status_code=400)
        
        prediction = Prediction()
        result = prediction.execute_pipeline(user_input)
        return JSONResponse(content={
            "status": "success",
            "prediction": result["prediction"],
            "probability": result["probability"],
            "threshold": result["threshold"],
            "preprocessing_steps": result["preprocessing_steps"],
            "token_sequence": result["sequence"]
            })

    except Exception as e:

        error_msg = str(e)

        # Makes error messages more user-friendly
        if "No URLs matched" in error_msg or "gsutil" in error_msg:
            error_msg = "Could not connect to Google Cloud Storage. Please check your credentials."
        elif "GOOGLE_APPLICATION_CREDENTIALS" in error_msg:
            error_msg = "Google Cloud credentials not configured. Please set up authentication."
        elif not error_msg:
            error_msg = "Training completed but the new model did not outperform the existing one."
        
        return JSONResponse(content={"status": "error", "message": error_msg}, status_code=500)


# Run the server when executing this file directly
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)