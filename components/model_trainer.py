# Importing required modules
import os
import sys

# For handling and manipulating dataset
import pandas as pd

# For splitting dataset
from sklearn.model_selection import train_test_split

# For Deep Learning
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from logger import logging
from exception import CustomException
from constants import *
from entity.config_entity import ModelTrainerConfig
from entity.artifact_entity import DataTransformationArtifacts
from entity.artifact_entity import ModelTrainerArtifacts
from components.model import ModelArchitecture
from optimizer.nvidia_optimizer import optimize_gpu

class ModelTrainer:
    """
    Model Training Pipeline. Splits data into train and test split -> Tokenizes text into sequence -> Trains the LSTM model -> Saves the trained model
    """

    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config
        
        # Sets up GPU optimization before training
        optimize_gpu()

    def data_splitting(self, dataset):
        """
        Splits the data into train and test sets
        """

        try:
            logging.info("Carring out train-test-split operation...")
            logging.info("Reading data...")
            df = pd.read_csv(dataset, index_col = False)

            # Drops rows with missing values
            df = df.dropna(subset= [CONTENT, LABEL])

            # Makes sure Content column is all strings
            df[CONTENT] = df[CONTENT].astype(str)
            
            # Assigning descriptive and target features
            x = df[CONTENT]
            y = df[LABEL]

            logging.info("Splitting data into descriptive and target feature")
            
            # Train-test-split feature
            x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = RANDOM_STATE)

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
        

    def initiate_model_training(self):
        """
        Initiates model training
        """

        try:
            # Loading and splitting the dataset
            x_train, x_test, y_train, y_test = self.data_splitting(dataset = self.data_transformation_artifacts.transformed_data_path)

            # Building the LSTM model
            model_architecture = ModelArchitecture()
            model = model_architecture.get_model()

            logging.info(f"x_train size: {x_train.shape}")
            logging.info(f"x_test shape: {x_test.shape}")

            # Tokenizing the training data
            sequences_matrix, tokenizer = self.tokenization(x_train = x_train)

            # Training the model
            logging.info(f"Starting model training...")

            model.fit(
                sequences_matrix,
                y_train,
                batch_size= self.model_trainer_config.BATCH_SIZE,
                epochs= self.model_trainer_config.EPOCH,
                validation_split= self.model_trainer_config.SPLIT_SIZE
                )
            
            logging.info(f"Completed model training.")

            # Saving the tokenizer (For using it to convert new text into same number format)
            with open('tokenizer.pickle', 'wb') as pickle_file:
                pickle.dump(tokenizer, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

            # Creating output directory
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok= True)

            logging.info("Saving the trained model...")
            # Saving model to make prediction on the saved sequences
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)

            # Saving the testing and training data for reference
            x_test.to_csv(self.model_trainer_config.DF_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.TF_TEST_DATA_PATH)

            x_train.to_csv(self.model_trainer_config.DF_TRAIN_DATA_PATH)
            logging.info("Saved the trained model.")

            # Packaging all into an artifact object
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path= self.model_trainer_config.TRAINED_MODEL_PATH,
                df_test_path = self.model_trainer_config.DF_TEST_DATA_PATH,
                tf_test_path = self.model_trainer_config.TF_TEST_DATA_PATH
                )
            
            logging.info("Returning model artifacts...")

            return model_trainer_artifacts

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e