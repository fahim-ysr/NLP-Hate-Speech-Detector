# Importing required modules

import os
from datetime import datetime

# Common constraints
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = "hatespeechdataset"
DATASETS_ZIP = "Datasets.zip"
CONTENT = "Content"
LABEL = "Label"

# Data Ingestion Constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATASET1 = "HateSpeechDatasetBalanced.csv"
DATASET2 = "EmojiHate.csv"