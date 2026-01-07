import pytest
import sys
import os
from fastapi.testclient import TestClient
from application import app
from constants import MAX_WORDS, MAX_LENGTH, BUCKET_NAME, THRESHOLD, MODEL_NAME

# Adds project root path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API TESTS

# CONSTANTS VALIDATION

# MODEL ARCHITECTURE TEST

# DATA TRANSFORMATION TEST

# PIPELINE IMPORT TEST