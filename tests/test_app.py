import pytest
import sys
import os
from constants import MAX_WORDS, MAX_LENGTH, BUCKET_NAME, THRESHOLD, MODEL_NAME

# Adds project root path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fixture: Reusable test component
@pytest.fixture
def client():
    """Creates a resuable test client"""
    from fastapi.testclient import TestClient
    from application import app
    return TestClient(app)


# API TESTS
class TestAPIEndpoints:
    """Test for all API endpoints to ensure web server works correctly"""

    def test_home_page_returns_200(self, client):
        """Tests if homepage loads successfully"""
        response = client.get("/")
        assert response.status_code == 200

    def test_home_page_returns_html(self, client):
        """Verifies if homepage returns HTML content"""
        response = client.get("/")
        assert "text/html" in response.headers.get("content-type", "")

    def test_api_docs_accessible(self, client):
        """FastAPI documentation must be accessible"""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_prediction_endpoint_exists(self, client):
        """Checks if prediction endpoint exists"""
        response = client.get("/predict")
        assert response.status_code != 404


# CONSTANTS VALIDATION
class TestConstants:
    """Validates configuration constants to make sure they are set up correctly"""

    def text_max_words_valid(self):
        """MAX_WORDS must be a positive int"""
        assert isinstance(MAX_WORDS, int) and MAX_WORDS > 0

    def test_max_length_valid(self):
        """MAX_LENGTH must be positive int"""
        assert isinstance(MAX_LENGTH, int) and MAX_LENGTH > 0

    def test_threshold_range(self):
        """THRESHOLD must be between 0 and 1"""
        assert 0 < THRESHOLD < 1

    def test_bucket_config(self):
        """Google Cloud bucket must be configured"""
        assert BUCKET_NAME and len(BUCKET_NAME) > 0

    def test_model_format(self):
        """Model outputted must be .h5 format"""
        assert MODEL_NAME.endswith(".h5")


# DATA TRANSFORMATION TEST
class TestDataTransformation:
    """Tests text processing operation"""

    @pytest.fixture
    def transformer(self):
        """Creates transformer instance"""
        from components.data_transformation import DataTransformation
        from entity.config_entity import DataTransformationConfig
        from entity.artifact_entity import DataIngestionArtifacts
        # Creates mock artifacts with dummy paths
        mock_artifacts = DataIngestionArtifacts(dataset1="dummy_path1.csv", dataset2="dummy_path2.csv")
        return DataTransformation(data_transformation_config=DataTransformationConfig(), data_ingestion_artifacts=mock_artifacts)
    
    def test_data_format_returns_tuple(self, transformer):
        """Preprocessing must return (text, steps) tuple"""
        inp = transformer.data_format("Hello World!")
        assert isinstance(inp, tuple) and len(inp) == 2

    def test_handles_empty_input(self, transformer):
        """Must handle empty strings"""
        inp = transformer.data_format("")
        assert inp is not None


# PIPELINE IMPORT TEST
class TestPipelineImports:
    """Verifies all pipeline modules can be imported"""

    def test_training_pipeline(self):
        """Training pipeline must be importable"""
        from pipelines.train import Training
        assert Training is not None

    def test_prediction_pipeline(self):
        """Prediction pipeline must be importable"""
        from pipelines.prediction import Prediction
        assert Prediction is not None


# SMOKE TEST
class TestSmoke:
    """Fast test that verifies basic system functionality"""

    def test_python_version(self):
        """Must run on Python 3.8+"""
        assert sys.version_info >= (3, 8)

    def test_critical_imports(self):
        """All mandatory packages must be installed"""
        import fastapi
        import tensorflow
        import nltk
        assert True