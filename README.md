# Machine Learning Coursework

This repository contains the coursework reports, notebooks, generated report
assets, and the FastAPI deployment package in `cloud-deployment`.

## Prediction API

### Live Cloud Run service

The deployed Cloud Run service (`q1-success-api`) is live at
[https://q1-success-api-608121463228.asia-south1.run.app](https://q1-success-api-608121463228.asia-south1.run.app).
The container includes custom ASGI middleware that:

1. Rejects HTTP request payloads larger than **65,536 bytes (64 KiB)**.
2. Limits inference traffic to **60 prediction requests per 60 seconds** per
   instance and returns HTTP 429 when the limit is exceeded.

Check the public health endpoint:

```bash
curl --fail-with-body \
  --request GET \
  --url https://q1-success-api-608121463228.asia-south1.run.app/health
```

Retrieve the deployed model metadata:

```bash
curl --fail-with-body \
  --request GET \
  --url https://q1-success-api-608121463228.asia-south1.run.app/model-info
```

Submit the included prediction request from the repository root:

```bash
curl --fail-with-body \
  --request POST \
  --url https://q1-success-api-608121463228.asia-south1.run.app/predict \
  --header "Content-Type: application/json" \
  --data @cloud-deployment/predict_request.json
```

Unauthenticated requests to the `/health` and `/model-info` endpoints return
HTTP 200, confirming that the cloud service is operational.

### Local service

Start the API locally from the repository root:

```bash
docker build -t q1-success-api cloud-deployment
docker run --rm -p 8080:8080 q1-success-api
```

In another terminal, submit the included example request:

```bash
curl --fail-with-body \
  --request POST \
  --url http://localhost:8080/predict \
  --header "Content-Type: application/json" \
  --data @cloud-deployment/predict_request.json
```

> **Schema note:** The included example uses descriptive engineered feature
> names. The current `q1-v1` model accepts raw questionnaire fields named
> `Q6` through `Q34`, as defined in `cloud-deployment/feature_schema.json`.
> Consequently, the current API returns HTTP 422 for this example until the
> API/model contract is updated to support these descriptive fields.

## Main coursework files

- `Final Coursework Report.pdf` — final rendered submission
- `Final Coursework Report.md` — report source
- `ML-q1.ipynb` and `ML_q2.ipynb` — analysis notebooks
- `Question 1 Report.md`, `Question 2 Report.md`, and `Question 3 Report.md` — individual reports
- `cloud-deployment/` — API, model, tests, and deployment configuration
