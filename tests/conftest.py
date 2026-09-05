import pickle

import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.config import settings


@pytest.fixture
def sample_features() -> dict[str, object]:
    """Sample valid features for prediction tests."""
    return {
        "PU_DO": "74_42",
        "trip_distance": 3.2,
    }


@pytest.fixture(scope="session")
def trained_model():
    """Load the trained model once for the entire test session."""

    with settings.model_path.open("rb") as file:
        vectorizer, model = pickle.load(file)

    return vectorizer, model


@pytest.fixture
def client():
    """Create a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client
