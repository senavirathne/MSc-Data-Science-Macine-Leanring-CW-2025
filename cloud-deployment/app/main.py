from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json


# ============================================================
# Application configuration
# ============================================================

MODEL_NAME = "student-burnout-risk-classifier"
MODEL_VERSION = "v1"

MODEL_PATH = "model_package/model.joblib"
LABEL_ENCODER_PATH = "model_package/label_encoder.joblib"
SCHEMA_PATH = "model_package/feature_schema.json"
METRICS_PATH = "model_package/metrics.json"


# ============================================================
# Load model package
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        schema = json.load(file)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

except Exception as exc:
    raise RuntimeError(f"Failed to load model package: {exc}") from exc


expected_features = schema.get("features", [])

if not expected_features:
    raise RuntimeError(
        "No features were found in model_package/feature_schema.json"
    )


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Student Burnout Risk API",
    description=(
        "Private API for student burnout-risk classification. "
        "Predictions are intended for authorised human review only."
    ),
    version=MODEL_VERSION,
)


# ============================================================
# Request models
# ============================================================

class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="One student record containing all required model features.",
    )


class BatchPredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="A non-empty list of student records.",
    )


# ============================================================
# Helper functions
# ============================================================

def utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def validate_record_columns(record: dict[str, Any]) -> None:
    """Validate that one record contains all required features."""
    missing = [feature for feature in expected_features if feature not in record]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing required features",
                "missing": missing,
            },
        )


def prepare_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Validate records and return a DataFrame in the exact feature order
    expected by the trained model.
    """
    for index, record in enumerate(records):
        missing = [
            feature for feature in expected_features
            if feature not in record
        ]

        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Missing required features",
                    "record_index": index,
                    "missing": missing,
                },
            )

    dataframe = pd.DataFrame(records)

    try:
        return dataframe[expected_features]
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Input schema does not match the model schema",
                "details": str(exc),
            },
        ) from exc


def decode_predictions(predictions: Any) -> list[str]:
    """
    Convert encoded model predictions into the original class labels.
    """
    try:
        prediction_ids = [int(value) for value in predictions]
        decoded = label_encoder.inverse_transform(prediction_ids)
        return [str(label) for label in decoded]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to decode model predictions",
                "details": str(exc),
            },
        ) from exc


def build_probability_output(probability_row: Any) -> dict[str, float]:
    """
    Convert one predict_proba row into a class-to-probability mapping.
    """
    classes = [str(label) for label in label_encoder.classes_]

    return {
        class_name: float(probability_row[index])
        for index, class_name in enumerate(classes)
    }


# ============================================================
# API endpoints
# ============================================================

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": MODEL_NAME,
        "version": MODEL_VERSION,
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "timestamp": utc_timestamp(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": schema.get("target"),
        "problem_type": schema.get("problem_type"),
        "required_features": expected_features,
        "classes": [str(label) for label in label_encoder.classes_],
        "metrics": metrics,
        "timestamp": utc_timestamp(),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    validate_record_columns(request.features)

    try:
        input_data = pd.DataFrame(
            [request.features],
            columns=expected_features,
        )

        raw_prediction = model.predict(input_data)
        prediction_label = decode_predictions(raw_prediction)[0]

        response: dict[str, Any] = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "prediction": prediction_label,
            "timestamp": utc_timestamp(),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            response["probabilities"] = build_probability_output(probabilities)

        # Privacy-safe operational logging.
        # Do not print raw student features or personal identifiers.
        print(
            {
                "event": "prediction_completed",
                "model_version": MODEL_VERSION,
                "status": "success",
                "timestamp": response["timestamp"],
            }
        )

        return response

    except HTTPException:
        raise
    except Exception as exc:
        print(
            {
                "event": "prediction_failed",
                "model_version": MODEL_VERSION,
                "status": "error",
                "timestamp": utc_timestamp(),
                "error_type": type(exc).__name__,
            }
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed",
                "details": str(exc),
            },
        ) from exc


@app.post("/batch-predict")
def batch_predict(request: BatchPredictRequest) -> dict[str, Any]:
    input_data = prepare_dataframe(request.records)

    try:
        raw_predictions = model.predict(input_data)
        prediction_labels = decode_predictions(raw_predictions)

        probability_matrix = None

        if hasattr(model, "predict_proba"):
            probability_matrix = model.predict_proba(input_data)

        results: list[dict[str, Any]] = []

        for index, prediction_label in enumerate(prediction_labels):
            result: dict[str, Any] = {
                "record_index": index,
                "model_version": MODEL_VERSION,
                "prediction": prediction_label,
            }

            if probability_matrix is not None:
                result["probabilities"] = build_probability_output(
                    probability_matrix[index]
                )

            results.append(result)

        timestamp = utc_timestamp()

        print(
            {
                "event": "batch_prediction_completed",
                "model_version": MODEL_VERSION,
                "batch_size": len(results),
                "status": "success",
                "timestamp": timestamp,
            }
        )

        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "batch_size": len(results),
            "results": results,
            "timestamp": timestamp,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(
            {
                "event": "batch_prediction_failed",
                "model_version": MODEL_VERSION,
                "batch_size": len(request.records),
                "status": "error",
                "timestamp": utc_timestamp(),
                "error_type": type(exc).__name__,
            }
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Batch prediction failed",
                "details": str(exc),
            },
        ) from exc
