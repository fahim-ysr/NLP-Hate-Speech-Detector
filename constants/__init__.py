# Importing required modules

import os
from datetime import datetime

# Common Constants
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

# Data Transformation Constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
TRANSFORMED_FILE = "transformed.csv"
DATA_DIR = "data"

# Model Architecture Constants
MAX_WORDS = 50000
MAX_LENGTH = 300
LOSS = "binary_crossentrophy"
METRICS = ["accuracy"]
ACTIVATION = "sigmoid"

# Model Training Constants
MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
TRAINED_MODEL_DIR = "trained_model"
TRAINED_MODEL_NAME = "model.h5"
DF_TEST_FILE_NAME = "df_test.csv"
TF_TEST_FILE_NAME = "tf_test.csv"
DF_TRAIN_FILE_NAME = "df_train.csv"
RANDOM_STATE = 23
EPOCH = 1
BATCH_SIZE = 128
SPLIT_SIZE = 0.2