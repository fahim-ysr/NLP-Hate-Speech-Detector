# Importing required modules

import os
import sys
from zipfile import ZipFile
from logger import logging
from exception import CustomException
from config.gcloud_sync import GCloudSync
from entity.config_entity import DataIngestionConfig

class DataIngestion:
    
    # Constructor function
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = GCloudSync()

    def get_data(self):
        try:
            logging.info("Running get_data() method from DataIngestion class...")
            # Creates data ingestion artifact directory
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok= True)

            # Calls the sync_from_gcloud() method
            self.gcloud.sync_from_gcloud(self.data_ingestion_config.BUCKET_NAME, self.data_ingestion_config.DATASETS_ZIP)

        except Exception as e:
            raise CustomException(e, sys) from e