# Importing required modules

import sys
import os
# For splitting dataset
from sklearn.model_selection import train_test_split
# For handling and manipulating dataset
import pandas as pd
import seaborn as sns
# For text processing
import re
import nltk
import string
from nltk.corpus import stopwords

from logger import logging
from exception import CustomException
from entity.config_entity import DataTransformationConfig
from entity.artifact_entity import DataIngestionArtifacts
from entity.artifact_entity import DataTransformationArtifacts


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts


    def data_clean(self):
        """
        Cleans both datasets and returns them as concatanated in dataframe format
        """

        try:
            logging.info("Initiating data cleaning...")

            # Importing both the datasets
            df1 = pd.read_csv(self.data_ingestion_artifacts.dataset1)
            df2 = pd.read_csv(self.data_ingestion_artifacts.dataset2)

            # Data cleaning: Dropping unnecessary fields from dataset 2
            unnecessary_fields = ["entry_id", "type", "target", "round.base", "round.set", "set", "split", "matched_text", "matched_id"]
            df2.drop(columns = unnecessary_fields, axis= 1, inplace = True)

            # Renames fields of dataset 2 to match dataset 1
            df2.rename(columns = {"text":"Content", "label_gold":"Label"}, inplace = True)

            # Merges both cleaned datasets
            df = pd.concat([df1, df2], axis = 0, ignore_index=True)

            logging.info("Completed data cleaning.")
            return df

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
        
    
    def data_format(self, text):
        """
        Formats data in a dataset. Removes URLs, punctuations, numbers and stopwords. Also applies stemming to reduce words to their root form.
        """

        try:
            logging.info("Initiating data formatting...")

            # Initializing stemmers and stopwords on the data
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))

            # Formatting data
            text = str(text).lower()
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'<.*?>+', '', text)
            text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub(r'\n', '', text)
            text = re.sub(r'\w*\d\w*', '', text)

            # Filters out common words that do not add meaning (for e.g. articles)
            text = [word for word in text.split(' ') if word not in stopword]
            text=" ".join(text)

            # Applying stemmers to reduces words to root form (for e.g. Running -> Run)
            text = [stemmer.stem(word) for word in text.split(' ')]
            text=" ".join(text)

            logging.info("Completed data formatting.")

            return text

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_data_transformation(self):
        """
        Initiates data transformation.
        """

        try:   
            logging.info("Initiating data transformation...")

            # Cleans the datasets and then concatenates the datasets. Returns one concatenated dataset
            df = self.data_clean()
            
            # Formats data in the concatenated dataset
            df[self.data_transformation_config.CONTENT] = df[self.data_transformation_config.CONTENT].apply(self.data_format)

            # Makes a directory and saves the concatenated dataset as CSV
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok= True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index= False, header= True)

            # Creates data transformation artifact
            data_transformation_artifact = DataTransformationArtifacts(transformed_data_path = self.data_transformation_config.TRANSFORMED_FILE_PATH)

            logging.info("Completed data transformation.")
            return data_transformation_artifact

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e