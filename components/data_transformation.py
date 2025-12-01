# Importing required modules

import sys
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

    def data_cleaning(self):
        """
        Cleans both datasets and returns them as concatanated in dataframe format
        """

        try:
            logging.info("Initiating data cleaning...")

            # Importing both the datasets
            df1 = pd.read_csv(self.data_ingestion_artifacts.dataset1)
            df2 = pd.read_csv(self.data_ingestion_artifacts.dataset2)

            # Dropping unnecessary fields from dataset 2
            unnecessary_fields = ["entry_id", "type", "target", "round.base", "round.set", "set", "split", "matched_text", "matched_id"]
            df2.drop(columns = unnecessary_fields, axis= 1, inplace = True)

            # Merges both cleaned datasets
            df = pd.concat([df1, df2], axis = 0, ignore_index=True)

            return df

        except Exception as e:
            raise CustomException(e, sys) from e