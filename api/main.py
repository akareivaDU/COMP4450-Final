"""FastAPI service that serves NYC taxi fare predictions and logs them to DynamoDB."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException

from api import model_loader
from api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from common import db
from common.features import MIN_FARE, build_features_row, fare_bucket

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("taxi-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model at startup so the first request is not slow."""
    model_loader.try_load()
    yield


app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="Predicts yellow taxi fares and logs every prediction for monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "taxi-fare-api", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = model_loader.try_load()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=model_loader.model_version(),
        database_logging=db.enabled(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        bundle = model_loader.load_model()
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as 503
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    started = time.perf_counter()
    pickup = request.pickup_datetime or datetime.now(UTC)
    features = build_features_row(
        trip_distance=request.trip_distance,
        passenger_count=request.passenger_count,
        pickup_datetime=pickup,
        pu_location_id=request.pu_location_id,
        do_location_id=request.do_location_id,
    )
    raw_prediction = float(bundle["model"].predict(features)[0])
    predicted_fare = round(max(raw_prediction, MIN_FARE), 2)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    prediction_id = str(uuid.uuid4())
    bucket = fare_bucket(predicted_fare)
    record = {
        "prediction_id": prediction_id,
        "created_at": db.utc_now_iso(),
        "pickup_datetime": pickup.isoformat(),
        "trip_distance": request.trip_distance,
        "passenger_count": request.passenger_count,
        "pu_location_id": request.pu_location_id,
        "do_location_id": request.do_location_id,
        "predicted_fare": predicted_fare,
        "fare_bucket": bucket,
        "latency_ms": latency_ms,
        "model_version": bundle.get("model_version", "unknown"),
    }

    logged = False
    try:
        logged = db.log_prediction(record)
    except Exception as exc:  # noqa: BLE001 - a logging failure must not fail the prediction
        logger.error("Failed to write prediction log: %s", exc)

    return PredictionResponse(
        prediction_id=prediction_id,
        predicted_fare=predicted_fare,
        fare_bucket=bucket,
        model_version=bundle.get("model_version"),
        latency_ms=latency_ms,
        logged_to_db=logged,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Attach the real fare to a previous prediction so live accuracy can be measured."""
    try:
        recorded = db.record_feedback(request.prediction_id, request.actual_fare)
    except db.PredictionNotFound as exc:
        raise HTTPException(status_code=404, detail="Unknown prediction_id") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc

    message = "Feedback recorded" if recorded else "Database logging is disabled"
    return FeedbackResponse(
        prediction_id=request.prediction_id, recorded=recorded, message=message
    )
