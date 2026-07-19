import pytest
from fastapi.testclient import TestClient

from app import app, score


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_matches_packaged_metadata(model_metadata: dict) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json() == model_metadata
    assert response.json()["target_source"] == "Q35a"
    assert response.json()["selected_model"] == "Logistic Regression"


def test_predict_returns_expected_binary_result(minimal_valid_features: dict) -> None:
    expected = score([minimal_valid_features])[0]

    response = client.post("/predict", json={"features": minimal_valid_features})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == expected["prediction"]
    assert body["positive_class_probability"] == pytest.approx(
        expected["positive_class_probability"]
    )
    assert body["model_version"] == "q1-v1"
    assert 0.0 <= body["positive_class_probability"] <= 1.0


def test_predict_rejects_empty_features() -> None:
    response = client.post("/predict", json={"features": {}})

    assert response.status_code == 422
    assert "non-empty feature dictionary" in response.json()["detail"]


def test_predict_requires_features_object() -> None:
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_rejects_unknown_questionnaire_field(
    minimal_valid_features: dict,
) -> None:
    invalid = {**minimal_valid_features, "student_id": 12345}

    response = client.post("/predict", json={"features": invalid})

    assert response.status_code == 422
    assert "Unknown questionnaire fields" in response.json()["detail"]
    assert "student_id" in response.json()["detail"]
