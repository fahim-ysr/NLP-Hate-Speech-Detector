# Importing required modules

# For handling and manipulating dataset
import pandas as pd
import seaborn as sns

# For splitting dataset
from sklearn.model_selection import train_test_split

# For Deep Learning
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from logger import logging
from exception import CustomException
from constants import *
from entity.config_entity import ModelTrainerConfig
from entity.artifact_entity import DataIngestionArtifacts
from entity.artifact_entity import DataTransformationArtifacts
from components.model import ModelArchitecture

class ModelTrainer:

    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def data_splitting(self, dataset):
        """
        Carries out train-test-split with the given dataset
        """
        pass