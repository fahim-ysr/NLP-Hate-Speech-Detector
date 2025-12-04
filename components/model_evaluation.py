# Importing required modules
import os
import sys

# For handling and manipulating dataset
import pandas as pd
import seaborn as sns

# For splitting dataset
from sklearn.model_selection import train_test_split

# For Deep Learning
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

from logger import logging
from exception import CustomException
from constants import *
from entity.config_entity import ModelTrainerConfig
from entity.config_entity import ModelEvaluationConfig
from config.gcloud_sync import GCloudSync
from entity.artifact_entity import DataIngestionArtifacts
from entity.artifact_entity import DataTransformationArtifacts
from entity.artifact_entity import ModelTrainerArtifacts
from components.model import ModelArchitecture

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