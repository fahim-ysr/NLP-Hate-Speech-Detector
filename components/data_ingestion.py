# Importing required modules

import os
import sys
from zipfile import ZipFile
from logger import logging
from exception import CustomException
from config.gcloud_sync import GCloudSync
from entity.config_entity import DataIngestionConfig
from entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    """
    Handles downloading and extracting of dataset from Google Cloud Storage
    """
    
    # Constructor function
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = GCloudSync()

    def get_data(self):
        """
        Downloads the dataset zip file from Google Cloud Storage
        """

        try:
            logging.info("Running get_data() method from DataIngestion class...")
            # Creates data ingestion artifact directory
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok= True)

            # Calls the sync_from_gcloud() method
            self.gcloud.sync_from_gcloud(self.data_ingestion_config.BUCKET_NAME, self.data_ingestion_config.DATASETS_ZIP, self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
            logging.info("Completed get_data() method execution from DataIngestion class.")

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def unzip_and_clean(self):
        """
        Extracts the downloaded zip file and returns the path to both datasets
        """

        logging.info("Unzipping and cleaning datasets...")

        try:
            # Unzips the datasets
            with ZipFile(self.data_ingestion_config.DATASETS_ZIP_PATH, 'r') as file:
                file.extractall(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)

            logging.info("Successfully unzipped datasets.")

            # Returns the paths of both datasets
            dataset1 = self.data_ingestion_config.DATA_ARTIFACTS_DIR
            dataset2 = self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR

            return dataset1, dataset2

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def data_ingestion(self):
        """
        Automates data ingestion from process: Google Cloud Storage -> Unzip and clean -> Returns path
        """

        logging.info("Initiating data ingestion...")

        try:
            self.get_data()
            logging.info("Fetched data from GCloud.")
            dataset1, dataset2 = self.unzip_and_clean()
            logging.info("Unzipped and cleaned the datasets.")

            data_ingestion_artifacts = DataIngestionArtifacts(dataset1, dataset2)

            logging.info(f"Data ingestion artifacts: {data_ingestion_artifacts}.")

            return data_ingestion_artifacts
        
        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
