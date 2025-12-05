# Importing required modules

import os
import sys
import io

import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from logger import logging
from exception import CustomException
from constants import *
from components.data_transformation import DataTransformation
from entity.artifact_entity import DataIngestionArtifacts
from entity.config_entity import DataTransformationConfig
from config.gcloud_sync import GCloudSync


class Prediction:
    """
    Prediction Pipeline. Carries out real-time prediction based on user input
    """

    def __init__(self):
        # Google Cloud bucket where trained model is stored
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME

        # Data transformation to clean user input the same way training data was cleaned
        self.data_transformation = DataTransformation(data_transformation_config= DataTransformationConfig, data_ingestion_artifacts= DataIngestionArtifacts)

        # Local path to store downloaded model
        self.model_path = os.path.join("artifacts", "PredictionModel")
        self.gcloud = GCloudSync()

    
    def fetch_model_from_gcloud(self):
        """
        Downloads the best performing model from Google Cloud Storage
        """

        logging.info("Fetching model from Google Cloud Storage...")

        try:
            # Makes directory before downloading
            os.makedirs(self.model_path, exist_ok= True)

            # Fetches the model from Google Cloud bucket
            self.gcloud.sync_from_gcloud(self.bucket_name, self.model_name, self.model_path)

            # Builds full path to downloaded model
            best_model_path = os.path.join(self.model_path, self.model_name)

            logging.info("Completed fetching model from Google Cloud Storage.")

            return best_model_path

        
        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def predict(self, best_model_path, text):
        """
        Takes in raw text input and returns "Hate Speech" or "No Hate". Cleans text -> Converts to sequences -> Runs it through the LSTM model
        """

        logging.info("Running the prediction function...")

        try:

            # Loads the trained LSTM model from local storage
            best_model_path = self.fetch_model_from_gcloud()
            load_model = keras.models.load_model(best_model_path)

            # Loads the tokenizer used during training (for maintaining consistent encoding)
            with open("tokenizer.pickle", "rb") as pickle_file:
                load_tokenizer = pickle.load(pickle_file)

            # Cleans the input text the same way training data was cleaned
            text = self.data_transformation.data_format(text)
            text = [text]
            print(text)

            # Converts words to numbers using tokenizer
            sequence = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(sequence, maxlen= MAX_LENGTH)
            print(sequence)

            # Runs prediction
            pred = load_model.predict(padded)
            pred

            print(f"Prediction: {pred}")

            # Anything above the threshold 0.3 is considered hate speech
            if pred > 0.3:
                output = "Hate Speech"
                print(output)
                return output

            else:
                output = "No Hate"
                return output


        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def execute_pipeline(self, text):

        logging.info("Prediction pipeline started...")

        try:
            # Gets the best model from cloud storage
            best_model_path = self.fetch_model_from_gcloud()
            # Runs prediction on user input
            predicted_text = self.predict(best_model_path, text)
            
            logging.info("Prediction pipeline ended.")

            return predicted_text


        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
