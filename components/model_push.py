# Importing required modules
import os
import sys

from logger import logging
from exception import CustomException
from config.gcloud_sync import GCloudSync
from entity.config_entity import ModelPushConfig
from entity.artifact_entity import ModelPushArtifacts


class ModelPush:
    def __init__(self, model_push_config: ModelPushConfig):
        """
        Sets up model pushing woth config and Google Cloud connection
        """

        self.model_push_config = model_push_config
        self.gcloud = GCloudSync()

    
    def initiate_model_push(self):
        """
        Uploads the trained model to Google Cloud Storage bucket
        """

        logging.info("Initiating model pushing to Google Cloud bucket...")

        try:
            # Uploads the model file to Google Cloud bucket
            self.gcloud.sync_to_gcloud(self.model_push_config.BUCKET_NAME, self.model_push_config.TRAINED_MODEL_PATH, self.model_push_config.MODEL_NAME)

            logging.info("Uploaded the best model to Google Cloud Storage.")

            # Saving the model
            model_push_artifacts = ModelPushArtifacts(bucket_name= self.model_push_config.BUCKET_NAME)

            # Outputs the artifact
            logging.info("Completed model pushing to Google Cloud bucket.")
            return model_push_artifacts


        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
