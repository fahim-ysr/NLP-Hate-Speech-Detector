from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    dataset1: str
    dataset2: str

@dataclass
class DataTransformationArtifacts:
    transformed_data_path :str