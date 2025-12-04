# Importing required modules

from dataclasses import dataclass
from constants import *
import os

@dataclass
class DataIngestionConfig:
    """
    Configuration for Data Ingestion. Defines where to download data from and where to store it locally
    """

    def __init__(self):
        self.BUCKET_NAME = BUCKET_NAME
        self.DATASETS_ZIP = DATASETS_ZIP
        self.DATA_INGESTION_ARTIFACTS_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_ARTIFACTS_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATASET1)
        self.NEW_DATA_ARTIFACTS_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATASET2)
        self.DATASETS_ZIP_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.DATASETS_ZIP_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,self.DATASETS_ZIP)

@dataclass
class DataTransformationConfig:
    """
    Configuration for Data Transformation. Defines where to save cleaned and preprocessed data
    """

    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TRANSFORMED_FILE)
        self.CONTENT = CONTENT
        self.LABEL = LABEL

@dataclass
class ModelTrainerConfig:
    """
    Configuration for Model Training. Defines model hyperparameters and output paths
    """

    def __init__(self):
        self.TRAINED_MODEL_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)
        self.DF_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, DF_TEST_FILE_NAME)
        self.TF_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, TF_TEST_FILE_NAME)
        self.DF_TRAIN_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, DF_TRAIN_FILE_NAME)
        self.MAX_WORDS = MAX_WORDS
        self.MAX_LENGTH = MAX_LENGTH