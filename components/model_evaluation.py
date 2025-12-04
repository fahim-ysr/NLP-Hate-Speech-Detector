# Importing required modules
import os
import sys

# For handling and manipulating dataset
import pandas as pd
import numpy as np

# For Deep Learning
import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
import pickle

from logger import logging
from exception import CustomException
from constants import *
from entity.config_entity import ModelEvaluationConfig
from config.gcloud_sync import GCloudSync
from entity.artifact_entity import DataTransformationArtifacts
from entity.artifact_entity import ModelTrainerArtifacts

class ModelEvaluation:
    """
    Docstring for ModelEvaluation
    """

    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifacts: ModelTrainerArtifacts, data_transformation_artifacts: DataTransformationArtifacts):
        """
        Docstring for __init__
        """

        # Config with paths and bucket information
        self.model_evaluation_config = model_evaluation_config
        # Artifacts from model training (path to newly trained model)
        self.model_trainer_artifacts = model_trainer_artifacts
        # Artifacts from data transformation (path to test data)
        self.data_transformation_artifacts = data_transformation_artifacts
        # GCloud connection for downloading the current best model
        self.gcloud = GCloudSync()


    def get_best_model(self):
        """
        Downloads the current best model from GCloud. Then the model is compared against the local one to see if it is improved.
        """

        try:
            logging.info("Selecting the best model...")
            # Creates directory to store the downloaded model
            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok= True)

            # Downloads the model from GCloud bucket
            self.gcloud.sync_from_gcloud(self.model_evaluation_config.BUCKET_NAME, self.model_evaluation_config.MODEL_NAME, self.model_evaluation_config.BEST_MODEL_DIR_PATH)

            # Builds the path to the downloaded model
            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH, self.model_evaluation_config.MODEL_NAME)

            logging.info("Completed selecting the best model.")

            return best_model_path


        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
        
    
    def evaluation(self):
        """
        Evaulates the trained model on test set. Loads the tokenized and model -> Runs prediction -> Outputs the confusion matrix
        """

        try:
            logging.info("Evaluating the model...")
            print(self.model_trainer_artifacts.df_test_path)

            # Loading test sets
            x_test = pd.read_csv(self.model_trainer_artifacts.df_test_path, index_col= 0)
            print(x_test)

            y_test = pd.read_csv(self.model_trainer_artifacts.tf_test_path, index_col= 0)

            # Loads the tokenizer we saved during training
            with open("tokenizer.pickle", "rb") as pickle_file:
                tokenizer = pickle.load(pickle_file)

            # Loads the trained model
            load_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            # Converts dataframe to series for preprocessing
            x_test = (x_test['Content'].astype("str")).squeeze()
            y_test = y_test.squeeze()

            # Converts text to sequences and pads them
            test_sequences = tokenizer.texts_to_seqience(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen = MAX_LENGTH)

            print(f"__________________{x_test.shape}__________________")
            print(f"__________________{y_test.shape}__________________")
            
            # Evaluates the model based on its accuracy
            accuracy = load_model.evaluate(test_sequences_matrix, y_test)

            logging.info(f"Test accuracy: {accuracy}")

            # Makes prediction on test data
            LSTM_pred = load_model.predict(test_sequences_matrix)
            res = []

            # Converts probabilities to binary predictions
            for pred in LSTM_pred:
                if pred[0] < 0.5:
                    res.append(0)
                else:
                    res.append(1)

            # Outputs the confusion matrix to see TP and FP values
            print(confusion_matrix(y_test, res))

            logging.info(f"Confusion matrix: {confusion_matrix(y_test, res)}")

            logging.info("Completed model evaluation.")

            return accuracy


        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e