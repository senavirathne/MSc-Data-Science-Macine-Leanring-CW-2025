import json
import os
from collections import deque
from math import ceil
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def positive_integer_setting(name: str, default: int) -> int:
    """Read a positive integer environment setting or fail during startup."""
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


MAX_REQUEST_BYTES = positive_integer_setting("MAX_REQUEST_BYTES", 65_536)
PREDICT_RATE_LIMIT = positive_integer_setting("PREDICT_RATE_LIMIT", 60)
RATE_LIMIT_WINDOW_SECONDS = positive_integer_setting(
    "RATE_LIMIT_WINDOW_SECONDS",
    60,
)


class RequestSizeLimitMiddleware:
    """Reject oversized HTTP bodies before invoking the downstream application."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive.")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int,
        detail: str,
    ) -> None:
        response = JSONResponse(status_code=status_code, content={"detail": detail})
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                await self.reject(
                    scope,
                    receive,
                    send,
                    400,
                    "Invalid Content-Length header.",
                )
                return
            if declared_length < 0:
                await self.reject(
                    scope,
                    receive,
                    send,
                    400,
                    "Invalid Content-Length header.",
                )
                return
            if declared_length > self.max_body_bytes:
                await self.reject(
                    scope,
                    receive,
                    send,
                    413,
                    f"Request body exceeds {self.max_body_bytes} bytes.",
                )
                return

        buffered_body = bytearray()

        while True:
            message = await receive()

            if message["type"] == "http.disconnect":
                return

            if message["type"] != "http.request":
                raise RuntimeError(
                    f"Unexpected ASGI request message: {message['type']}"
                )

            body_chunk = message.get("body", b"")
            if len(buffered_body) + len(body_chunk) > self.max_body_bytes:
                await self.reject(
                    scope,
                    receive,
                    send,
                    413,
                    f"Request body exceeds {self.max_body_bytes} bytes.",
                )
                return

            buffered_body.extend(body_chunk)
            if not message.get("more_body", False):
                break

        body_delivered = False

        async def replay_receive() -> Message:
            nonlocal body_delivered
            if not body_delivered:
                body_delivered = True
                return {
                    "type": "http.request",
                    "body": bytes(buffered_body),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay_receive, send)


class SlidingWindowRateLimitMiddleware:
    """Apply an in-memory sliding-window limit within one app instance."""

    def __init__(
        self,
        app: ASGIApp,
        request_limit: int,
        window_seconds: int,
        path: str = "/predict",
    ) -> None:
        if request_limit <= 0 or window_seconds <= 0:
            raise ValueError("Rate-limit settings must be positive.")
        self.app = app
        self.request_limit = request_limit
        self.window_seconds = window_seconds
        self.path = path
        self.request_times: deque[float] = deque()
        self.lock = Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"] == self.path
        ):
            await self.app(scope, receive, send)
            return

        retry_after: int | None = None
        remaining = 0

        with self.lock:
            now = monotonic()
            window_start = now - self.window_seconds
            while self.request_times and self.request_times[0] <= window_start:
                self.request_times.popleft()

            if len(self.request_times) >= self.request_limit:
                retry_after = max(
                    1,
                    ceil(self.window_seconds - (now - self.request_times[0])),
                )
            else:
                self.request_times.append(now)
                remaining = self.request_limit - len(self.request_times)

        if retry_after is not None:
            response = JSONResponse(
                status_code=429,
                content={"detail": "Prediction request rate limit exceeded."},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.request_limit),
                    "X-RateLimit-Remaining": "0",
                },
            )
            await response(scope, receive, send)
            return

        async def send_with_rate_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-ratelimit-limit", str(self.request_limit).encode()),
                        (b"x-ratelimit-remaining", str(remaining).encode()),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_rate_headers)


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
app.add_middleware(
    SlidingWindowRateLimitMiddleware,
    request_limit=PREDICT_RATE_LIMIT,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_bytes=MAX_REQUEST_BYTES,
)


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
    if not records or any(
        not isinstance(record, dict) or not record for record in records
    ):
        raise ValueError("Each request must contain a non-empty feature dictionary.")

    unknown = sorted(
        set().union(*(record.keys() for record in records)) - set(raw_feature_names)
    )
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
        model_frame[column] = (
            model_frame[column].map(normalize_nominal).astype(object)
        )
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
