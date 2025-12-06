# Importing required modules

import sys
from logger import logging
from exception import CustomException
from components.data_ingestion import DataIngestion
from entity.config_entity import DataIngestionConfig
from entity.artifact_entity import DataIngestionArtifacts
from entity.artifact_entity import DataTransformationArtifacts
from components.data_transformation import DataTransformation
from entity.config_entity import DataTransformationConfig
from components.model_trainer import ModelTrainer
from entity.config_entity import ModelTrainerConfig
from entity.artifact_entity import ModelTrainerArtifacts
from components.model_evaluation import ModelEvaluation
from entity.config_entity import ModelEvaluationConfig
from entity.artifact_entity import ModelEvaluationArtifacts
from components.model_push import ModelPush
from entity.config_entity import ModelPushConfig
from entity.artifact_entity import ModelPushArtifacts


class Training:
    """
    Training pipeline. Carries out: Data Ingestion -> Data Transformation -> Model Training
    """

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_push_config = ModelPushConfig()

    def start_data_ingestion(self):
        logging.info("")

        try:
            logging.info("Starting data ingestion operation...")
            # Calls the data ingestion operation from data_ingestion file
            data_ingestion = DataIngestion(data_ingestion_config= self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.data_ingestion()
            logging.info("Completed data ingestion operation.")
            return data_ingestion_artifacts
        
        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def start_data_transformation(self, data_ingestion_artifact = DataIngestionArtifacts):

        try:
            logging.info("Starting data transformation operation...")
            data_transformation = DataTransformation(data_transformation_config = self.data_transformation_config, data_ingestion_artifacts = data_ingestion_artifact)

            data_transformation_artifacts = data_transformation.initiate_data_transformation()

            logging.info("Completed data transformation operation.")
            return data_transformation_artifacts

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
        
        
    def start_model_training(self, data_transformation_artifacts: DataTransformationArtifacts):
        """
        Docstring for start_model_training
        """

        logging.info("Starting model training operation...")

        try:
            model_trainer = ModelTrainer(data_transformation_artifacts= data_transformation_artifacts, model_trainer_config= self.model_trainer_config)
            
            model_trainer_artifacts = model_trainer.initiate_model_training()

            logging.info("Completed model training operation.")
            return model_trainer_artifacts

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e        


    def start_model_evaluation(self, model_trainer_artifacts: ModelTrainerArtifacts, data_transformation_artifacts: DataTransformationArtifacts):
        """

        """

        logging.info("Starting model evaluation operation...")

        try:
            model_evaluation = ModelEvaluation(data_transformation_artifacts= data_transformation_artifacts, model_evaluation_config= self.model_evaluation_config, model_trainer_artifacts= model_trainer_artifacts)

            model_evaluation_artifacts = model_evaluation.initialize_model_evaluation()

            logging.info("Completed model evaluation operation.")
            return model_evaluation_artifacts
            

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def start_model_push(self):
        """
        Uploads the new model to Google Cloud Storage only if it is the best model compared to newly trained model
        """

        logging.info("Starting model pushing operation...")

        try:
            model_push = ModelPush(model_push_config= self.model_push_config)
            model_push_artifacts = model_push.initiate_model_push()

            logging.info("Completed model pushing operation.")

            return model_push_artifacts


        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e



        
    def execute_pipeline(self):
        logging.info("Training pipeline started...")

        try:
            # Gets raw dataset from Google Cloud Storage
            data_ingestion_artifacts = self.start_data_ingestion()
            # Cleans and preprocess the data
            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts)
            # Trains the LSTM model
            model_trainer_artifacts= self.start_model_training(data_transformation_artifacts)
            # Evaluates and compares with the best model
            model_evaluation_artifacts = self.start_model_evaluation(model_trainer_artifacts= model_trainer_artifacts, data_transformation_artifacts= data_transformation_artifacts)

            if not (model_evaluation_artifacts.accept):
                logging.info("Trained model did not outperform the existing model. Skipping upload...")
                return {"status": "not_improved", "message": "Training completed, but the new model did not outperform the existing one. Model was not updated."}
            
            model_push_artifacts = self.start_model_push()


            logging.info("Training pipeline ended.")
            return {"status": "success", "message": "Training completed successfully! New model has been uploaded."}

        # Exception handling
        except Exception as e:
            raise CustomException(e, sys) from e