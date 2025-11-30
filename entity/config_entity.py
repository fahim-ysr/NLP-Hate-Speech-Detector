# Importing required modules

from dataclasses import dataclass
from constants import *
import os

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME = BUCKET_NAME
        self.DATASET_ZIP = DATASETS_ZIP
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATASET1)
        self.NEW_DATA_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATASET2)
        self.DATASET_ZIP_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.DATASET_ZIP_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,self.DATASET_ZIP)