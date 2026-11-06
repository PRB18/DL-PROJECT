"""
tests/test_api.py
-----------------
Unit and integration tests for FastAPI backend service.
"""

import io
import pytest
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from app.backend.main import app, load_artifacts


@pytest.fixture(scope="module")
def client():
    load_artifacts()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "num_classes" in data
    assert data["model_loaded"] is True


def test_classes_endpoint(client: TestClient):
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) > 0


def test_predict_endpoint_valid_image(client: TestClient):
    # Create valid synthetic RGB image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "top_predictions" in data
    assert len(data["top_predictions"]) == 3
    assert "is_uncertain_or_non_leaf" in data
    assert "treatment_advice" in data
    assert "disclaimer" in data


def test_predict_endpoint_invalid_file(client: TestClient):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 400
