from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()
    assert body["model_name"] == "student-burnout-risk-classifier"
    assert body["target"] == "Burnout_Risk_Level"
    assert body["problem_type"] == "multiclass_classification"
    assert set(body["classes"]) == {"High", "Medium", "Low"}
    assert "metrics" in body
    assert "timestamp" in body


def test_predict_rejects_missing_features() -> None:
    response = client.post(
        "/predict",
        json={"features": {}},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["error"] == "Missing required features"
    assert len(body["missing"]) > 0


def test_batch_predict_rejects_missing_features() -> None:
    response = client.post(
        "/batch-predict",
        json={"records": [{}]},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["error"] == "Missing required features"
    assert len(body["missing"]) > 0