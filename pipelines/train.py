# Importing required modules

import sys
from logger import logging
from exception import CustomException
from components.data_ingestion import DataIngestion
from entity.config_entity import DataIngestionConfig
from entity.artifact_entity import DataIngestionArtifacts


class Training:
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self):
        logging.info("")

        try:
            logging.info("Fetching data from GCloud...")
            # Calls the data ingestion operation from data_ingestion file
            data_ingestion = DataIngestion(data_ingestion_config= self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.data_ingestion()
            logging.info("Completed data ingestion from GCloud.")
            return data_ingestion_artifacts
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def execute_pipeline(self):
        logging.info("Training pipeline started...")

        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            logging.info("Training pipeline ended.")

        except Exception as e:
            raise CustomException(e, sys) from e