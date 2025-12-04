# Importing required modules

import sys
from logger import logging
from exception import CustomException
from components.data_ingestion import DataIngestion
from entity.config_entity import DataIngestionConfig
from entity.artifact_entity import DataIngestionArtifacts
from components.data_transformation import DataTransformation
from entity.config_entity import DataTransformationConfig
from entity.artifact_entity import DataTransformationArtifacts


class Training:
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()

    def start_data_ingestion(self):
        logging.info("")

        try:
            logging.info("Fetching data from GCloud...")
            # Calls the data ingestion operation from data_ingestion file
            data_ingestion = DataIngestion(data_ingestion_config= self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.data_ingestion()
            logging.info("Completed data ingestion from GCloud.")
            return data_ingestion_artifacts
        
        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def start_data_transformation(self, data_ingestion_artifact = DataIngestionArtifacts):

        try:
            logging.info("Starting data transformation...")
            data_transformation = DataTransformation(data_transformation_config = self.data_transformation_config, data_ingestion_artifacts = data_ingestion_artifact)

            data_transformation_artifacts = data_transformation.initiate_data_transformation()

            logging.info("Completed data transformation.")
            return data_transformation_artifacts

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e

        
    def execute_pipeline(self):
        logging.info("Training pipeline started...")

        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_ingestion_artifacts = self.start_data_transformation(data_ingestion_artifacts)
            logging.info("Training pipeline ended.")

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e