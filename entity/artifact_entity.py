from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    """
    Defines output from Data Ingestion
    """

    dataset1: str
    dataset2: str

@dataclass
class DataTransformationArtifacts:
    """
    Defines output from Data Transformation
    """

    transformed_data_path :str

@dataclass
class ModelTrainerArtifacts:
    """
    Defines output from Model Training
    """

    trained_model_path: str
    df_test_path: str
    tf_test_path: str


@dataclass
class ModelEvaluationArtifacts:
    """
    Defined output from Model Evaluation
    """

    accept: bool


@dataclass
class ModelPushArtifacts:
    """
    Defines output from Model Pushing
    """

    bucket_name: str