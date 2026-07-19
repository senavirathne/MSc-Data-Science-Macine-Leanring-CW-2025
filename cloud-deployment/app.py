from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model.joblib")
metadata = json.loads((BASE_DIR / "metadata.json").read_text())
schema = json.loads((BASE_DIR / "feature_schema.json").read_text())
raw_feature_names = schema["raw_api_features"]
model_feature_names = schema["model_feature_names"]
nominal_features = schema["nominal_features"]
api_range_rules = schema["range_rules"]
minimum_non_missing_raw_fields = schema["minimum_non_missing_raw_fields"]
engineered_feature_rules = schema["engineered_features"]

app = FastAPI(title="Q1 Associative Classification API", version="1.0")

class PredictionRequest(BaseModel):
    features: dict[str, Any]

def api_row_mean(frame, columns, minimum_answered):
    block = frame[columns].apply(pd.to_numeric, errors="coerce")
    answered = block.notna().sum(axis=1)
    return block.mean(axis=1, skipna=True).where(answered >= minimum_answered)

def engineer_model_features(frame):
    result = frame.copy()
    task_rule = engineered_feature_rules["chatgpt_task_breadth_rate"]
    task_block = result[task_rule["source_columns"]].apply(
        pd.to_numeric, errors="coerce"
    )
    task_answered = task_block.notna().sum(axis=1)
    result["chatgpt_task_breadth_rate"] = (
        task_block.ge(3).sum(axis=1)
        .div(task_answered.replace(0, np.nan))
        .where(task_answered >= task_rule["minimum_answered"])
    )

    capability_rule = engineered_feature_rules["capability_mean"]
    result["capability_mean"] = api_row_mean(
        result,
        capability_rule["source_columns"],
        capability_rule["minimum_answered"],
    )

    ethical_rule = engineered_feature_rules["ethical_concern_mean"]
    result["ethical_concern_mean"] = api_row_mean(
        result,
        ethical_rule["source_columns"],
        ethical_rule["minimum_answered"],
    )

    emotion_rule = engineered_feature_rules["emotion_balance"]
    result["emotion_balance"] = (
        api_row_mean(
            result,
            emotion_rule["positive_columns"],
            emotion_rule["minimum_positive_answered"],
        )
        - api_row_mean(
            result,
            emotion_rule["negative_columns"],
            emotion_rule["minimum_negative_answered"],
        )
    )
    return result

def normalize_nominal(value):
    if pd.isna(value):
        return np.nan
    return str(int(float(value)))

def prepare_api_records(records):
    if not records or any(not isinstance(record, dict) or not record for record in records):
        raise ValueError("Each request must contain a non-empty feature dictionary.")

    unknown = sorted(set().union(*(record.keys() for record in records)) - set(raw_feature_names))
    if unknown:
        raise ValueError(f"Unknown questionnaire fields: {unknown}")

    frame = pd.DataFrame(records).reindex(columns=raw_feature_names)
    for column in raw_feature_names:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"{column} contains a non-numeric value.")
        frame[column] = numeric

    for column, bounds in api_range_rules.items():
        lower, upper = bounds
        invalid = frame[column].notna() & (
            ~frame[column].between(lower, upper)
            | (frame[column] % 1 != 0)
        )
        if invalid.any():
            raise ValueError(
                f"{column} must contain an integer from {lower} to {upper}."
            )

    coverage = frame.notna().sum(axis=1)
    if (coverage < minimum_non_missing_raw_fields).any():
        raise ValueError(
            f"At least {minimum_non_missing_raw_fields} recognised raw fields are required."
        )

    engineered = engineer_model_features(frame)
    model_frame = engineered.reindex(columns=model_feature_names)
    for column in nominal_features:
        model_frame[column] = model_frame[column].map(normalize_nominal).astype(object)
    return model_frame

def score(records):
    frame = prepare_api_records(records)
    probabilities = model.predict_proba(frame)[:, 1]
    threshold = metadata["decision_threshold"]
    predictions = (probabilities >= threshold).astype(int)
    return [
        {
            "prediction": metadata["class_labels"][str(int(prediction))],
            "positive_class_probability": float(probability),
            "model_version": metadata["model_version"],
        }
        for prediction, probability in zip(predictions, probabilities)
    ]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return metadata

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return score([request.features])[0]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
