import numpy as np
import pandas as pd
import pytest

from app import prepare_api_records


def test_minimum_valid_record_has_expected_model_columns(
    minimal_valid_features: dict,
    feature_schema: dict,
) -> None:
    prepared = prepare_api_records([minimal_valid_features])

    assert prepared.shape == (1, len(feature_schema["model_feature_names"]))
    assert prepared.columns.tolist() == feature_schema["model_feature_names"]


def test_nominal_fields_are_normalized_to_strings(
    minimal_valid_features: dict,
    feature_schema: dict,
) -> None:
    prepared = prepare_api_records([minimal_valid_features])

    supplied_nominal = set(minimal_valid_features) & set(feature_schema["nominal_features"])
    assert supplied_nominal
    for field in supplied_nominal:
        assert prepared.at[0, field] == str(minimal_valid_features[field])


def test_all_four_composites_are_recreated(
    composite_source_features: dict,
) -> None:
    prepared = prepare_api_records([composite_source_features])

    assert prepared.at[0, "chatgpt_task_breadth_rate"] == pytest.approx(0.5)
    assert prepared.at[0, "capability_mean"] == pytest.approx(3.0)
    assert prepared.at[0, "ethical_concern_mean"] == pytest.approx(4.0)
    assert prepared.at[0, "emotion_balance"] == pytest.approx(3.0)


def test_numeric_strings_are_accepted(minimal_valid_features: dict) -> None:
    string_values = {field: str(value) for field, value in minimal_valid_features.items()}

    prepared = prepare_api_records([string_values])

    assert prepared.shape[0] == 1


def test_non_numeric_value_is_rejected(minimal_valid_features: dict) -> None:
    field = next(iter(minimal_valid_features))
    invalid = {**minimal_valid_features, field: "unknown"}

    with pytest.raises(ValueError, match="contains a non-numeric value"):
        prepare_api_records([invalid])


def test_out_of_range_value_is_rejected(
    minimal_valid_features: dict,
    feature_schema: dict,
) -> None:
    field = next(iter(minimal_valid_features))
    upper_bound = feature_schema["range_rules"][field][1]
    invalid = {**minimal_valid_features, field: upper_bound + 1}

    with pytest.raises(ValueError, match="must contain an integer"):
        prepare_api_records([invalid])


def test_fractional_value_is_rejected(minimal_valid_features: dict) -> None:
    field = next(iter(minimal_valid_features))
    invalid = {**minimal_valid_features, field: minimal_valid_features[field] + 0.5}

    with pytest.raises(ValueError, match="must contain an integer"):
        prepare_api_records([invalid])


def test_insufficient_field_coverage_is_rejected(
    minimal_valid_features: dict,
    feature_schema: dict,
) -> None:
    required_count = feature_schema["minimum_non_missing_raw_fields"]
    incomplete = dict(list(minimal_valid_features.items())[: required_count - 1])

    with pytest.raises(ValueError, match=f"At least {required_count}"):
        prepare_api_records([incomplete])


def test_missing_composite_sources_remain_missing(minimal_valid_features: dict) -> None:
    prepared = prepare_api_records([minimal_valid_features])

    assert pd.isna(prepared.at[0, "chatgpt_task_breadth_rate"])
    assert pd.isna(prepared.at[0, "capability_mean"])
    assert pd.isna(prepared.at[0, "ethical_concern_mean"])
    assert np.isnan(prepared.at[0, "emotion_balance"])
