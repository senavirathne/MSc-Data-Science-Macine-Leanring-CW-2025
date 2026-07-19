import json
from pathlib import Path

import joblib
import numpy as np

from app import model, prepare_api_records


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_package_contains_required_artifacts() -> None:
    required_files = {
        "app.py",
        "model.joblib",
        "metadata.json",
        "feature_schema.json",
        "api_contract.json",
        "requirements.txt",
        "Dockerfile",
    }

    assert required_files <= {path.name for path in PROJECT_DIR.iterdir() if path.is_file()}


def test_schema_is_consistent(feature_schema: dict) -> None:
    raw_features = feature_schema["raw_api_features"]
    model_features = feature_schema["model_feature_names"]

    assert len(raw_features) == len(set(raw_features))
    assert len(model_features) == len(set(model_features))
    assert set(feature_schema["nominal_features"]) <= set(model_features)
    assert set(feature_schema["range_rules"]) == set(raw_features)
    assert "Q35a" not in raw_features
    assert "Q35a" not in model_features


def test_metadata_describes_binary_q1_model(model_metadata: dict) -> None:
    assert model_metadata["model_version"] == "q1-v1"
    assert model_metadata["target_source"] == "Q35a"
    assert model_metadata["positive_class"] == 1
    assert set(model_metadata["class_labels"]) == {"0", "1"}
    assert model_metadata["decision_threshold"] == 0.5


def test_model_supports_binary_probability_scoring() -> None:
    assert hasattr(model, "predict_proba")
    assert np.array_equal(model.classes_, np.array([0, 1]))


def test_reloaded_model_reproduces_probability(
    minimal_valid_features: dict,
) -> None:
    prepared = prepare_api_records([minimal_valid_features])
    reloaded_model = joblib.load(PROJECT_DIR / "model.joblib")

    expected = model.predict_proba(prepared)
    actual = reloaded_model.predict_proba(prepared)

    assert np.allclose(actual, expected)


def test_api_contract_matches_implemented_routes() -> None:
    contract = json.loads((PROJECT_DIR / "api_contract.json").read_text(encoding="utf-8"))

    assert {"GET /health", "GET /model-info", "POST /predict"} <= set(contract)
