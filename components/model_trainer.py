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

from logger import logging
from exception import CustomException
from constants import *
from entity.config_entity import ModelTrainerConfig
from entity.artifact_entity import DataIngestionArtifacts
from entity.artifact_entity import DataTransformationArtifacts
from components.model import ModelArchitecture

class ModelTrainer:
    """
    Model Training Pipeline. Splits data into train and test split -> Tokenizes text into sequence -> Trains the LSTM model -> Saves the trained model
    """

    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def data_splitting(self, dataset):
        """
        Splits the data into train and test sets
        """

        try:
            logging.info("Carring out train-test-split operation...")
            logging.info("Reading data...")
            df = pd.read(dataset, index_col = False)
            
            # Assigning descriptive and target features
            x = dataset[CONTENT]
            y = dataset[LABEL]

            logging.info("Splitting data into descriptive and target feature")
            
            # Train-test-split feature
            x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 23)

            print(f"Length of train set: {len(x_train)}, {len(y_train)}")
            print(f"Length of test set: {len(x_test)}, {len(y_test)}")

            logging.info("Completed train-test-split operation.")

            return x_train, x_test, y_train, y_test

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
        
    
    def tokenization(self, x_train):
        """
        Converts text into numerical sequences that LSTM model can understand
        Example: "I hate you" -> [45, 892, 23] -> [0, 0, ..., 45, 892, 23]
        """

        try:
            logging.info("Applying tokenization on data...")

            # Encodes words to numeric values
            tokenizer = Tokenizer(num_words = self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)

            # Converts words to sequence of integers
            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"Converting text to sequence: {sequences}")

            # Pads shorter sequences with 0 so all inputs are exactly 300 tokens long
            sequences_matrix = pad_sequences(sequences,maxlen= self.model_trainer_config.MAX_LENGTH)

            logging.info("Completed tokenization on data.")
            logging.info(f"Sequence matrix: {sequences_matrix}")

            return sequences_matrix, tokenizer

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e