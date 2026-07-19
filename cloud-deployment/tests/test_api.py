import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from app import (
    MAX_REQUEST_BYTES,
    PREDICT_RATE_LIMIT,
    RequestSizeLimitMiddleware,
    SlidingWindowRateLimitMiddleware,
    app,
    score,
)


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
    assert response.headers["x-ratelimit-limit"] == str(PREDICT_RATE_LIMIT)
    assert int(response.headers["x-ratelimit-remaining"]) < PREDICT_RATE_LIMIT


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


def test_declared_oversized_request_is_rejected_before_parsing() -> None:
    oversized_body = b"x" * (MAX_REQUEST_BYTES + 1)

    response = client.post(
        "/predict",
        content=oversized_body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": f"Request body exceeds {MAX_REQUEST_BYTES} bytes."
    }


def test_chunked_oversized_request_returns_413_through_fastapi() -> None:
    chunks = iter([b"x" * 40_000, b"x" * 40_000])

    response = client.post(
        "/predict",
        content=chunks,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": f"Request body exceeds {MAX_REQUEST_BYTES} bytes."
    }
    assert "x-ratelimit-limit" not in response.headers


def test_streamed_body_size_is_counted_without_content_length() -> None:
    downstream_called = False

    async def downstream(scope, receive, send) -> None:
        nonlocal downstream_called
        downstream_called = True
        await JSONResponse({"status": "unexpected"})(scope, receive, send)

    middleware = RequestSizeLimitMiddleware(downstream, max_body_bytes=10)
    incoming = iter(
        [
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"789012", "more_body": False},
        ]
    )
    outgoing = []

    async def receive():
        return next(incoming)

    async def send(message) -> None:
        outgoing.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/predict",
        "raw_path": b"/predict",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    asyncio.run(middleware(scope, receive, send))

    response_starts = [
        message for message in outgoing if message["type"] == "http.response.start"
    ]
    response_bodies = [
        message for message in outgoing if message["type"] == "http.response.body"
    ]
    assert downstream_called is False
    assert len(response_starts) == 1
    assert response_starts[0]["status"] == 413
    assert len(response_bodies) == 1


def test_exact_size_stream_is_replayed_to_downstream() -> None:
    consumed_body = bytearray()

    async def consume_body(scope, receive, send) -> None:
        while True:
            message = await receive()
            consumed_body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await JSONResponse({"status": "consumed"})(scope, receive, send)

    middleware = RequestSizeLimitMiddleware(consume_body, max_body_bytes=10)
    incoming = iter(
        [
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"7890", "more_body": False},
        ]
    )
    outgoing = []

    async def receive():
        return next(incoming)

    async def send(message) -> None:
        outgoing.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/predict",
        "raw_path": b"/predict",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    asyncio.run(middleware(scope, receive, send))

    response_start = next(
        message for message in outgoing if message["type"] == "http.response.start"
    )
    assert response_start["status"] == 200
    assert bytes(consumed_body) == b"1234567890"


def test_predict_rate_limit_returns_429_and_retry_after() -> None:
    limited_app = FastAPI()

    @limited_app.post("/predict")
    def limited_predict() -> dict[str, str]:
        return {"status": "ok"}

    limited_app.add_middleware(
        SlidingWindowRateLimitMiddleware,
        request_limit=2,
        window_seconds=60,
    )

    with TestClient(limited_app) as limited_client:
        first = limited_client.post("/predict")
        second = limited_client.post("/predict")
        rejected = limited_client.post("/predict")

    assert first.status_code == 200
    assert first.headers["x-ratelimit-remaining"] == "1"
    assert second.status_code == 200
    assert second.headers["x-ratelimit-remaining"] == "0"
    assert rejected.status_code == 429
    assert rejected.json() == {
        "detail": "Prediction request rate limit exceeded."
    }
    assert rejected.headers["retry-after"] == "60"
    assert rejected.headers["x-ratelimit-limit"] == "2"


def test_rate_limit_does_not_send_while_holding_thread_lock() -> None:
    class TrackingLock:
        def __init__(self) -> None:
            self.held = False

        def __enter__(self):
            assert self.held is False
            self.held = True
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self.held = False

    async def downstream(scope, receive, send) -> None:
        await JSONResponse({"status": "ok"})(scope, receive, send)

    middleware = SlidingWindowRateLimitMiddleware(
        downstream,
        request_limit=1,
        window_seconds=60,
    )
    tracking_lock = TrackingLock()
    middleware.lock = tracking_lock
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/predict",
        "raw_path": b"/predict",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    async def exercise_limit() -> list[dict]:
        outgoing = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message) -> None:
            assert tracking_lock.held is False
            outgoing.append(message)

        await middleware(scope, receive, send)
        await middleware(scope, receive, send)
        return outgoing

    outgoing = asyncio.run(exercise_limit())
    response_starts = [
        message for message in outgoing if message["type"] == "http.response.start"
    ]
    assert [message["status"] for message in response_starts] == [200, 429]
