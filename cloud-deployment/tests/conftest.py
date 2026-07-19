import json
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def feature_schema() -> dict:
    return json.loads((PROJECT_DIR / "feature_schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def model_metadata() -> dict:
    return json.loads((PROJECT_DIR / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture()
def minimal_valid_features(feature_schema: dict) -> dict[str, int]:
    """Return the minimum valid payload using each field's lower bound."""
    required_count = feature_schema["minimum_non_missing_raw_fields"]
    selected_fields = feature_schema["raw_api_features"][:required_count]
    range_rules = feature_schema["range_rules"]
    return {field: range_rules[field][0] for field in selected_fields}


@pytest.fixture()
def composite_source_features(feature_schema: dict) -> dict[str, int]:
    """Return sufficient source answers to calculate all four composites."""
    rules = feature_schema["engineered_features"]
    features: dict[str, int] = {}

    task_columns = rules["chatgpt_task_breadth_rate"]["source_columns"]
    features.update({field: 3 if index < 6 else 2 for index, field in enumerate(task_columns)})

    capability_columns = rules["capability_mean"]["source_columns"]
    capability_values = [1, 2, 3, 4, 5] * 2
    features.update(dict(zip(capability_columns, capability_values)))

    ethical_columns = rules["ethical_concern_mean"]["source_columns"]
    features.update({field: 4 for field in ethical_columns})

    emotion_rule = rules["emotion_balance"]
    features.update({field: 5 for field in emotion_rule["positive_columns"]})
    features.update({field: 2 for field in emotion_rule["negative_columns"]})
    return features
