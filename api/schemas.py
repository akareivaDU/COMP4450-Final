"""Request and response models for the fare prediction API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    trip_distance: float = Field(..., gt=0, le=100, description="Trip distance in miles")
    passenger_count: int = Field(1, ge=1, le=6)
    pu_location_id: int = Field(..., ge=1, le=265, description="TLC pickup zone id")
    do_location_id: int = Field(..., ge=1, le=265, description="TLC dropoff zone id")
    pickup_datetime: datetime | None = Field(
        None, description="Pickup time; defaults to the current UTC time"
    )


class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_fare: float
    fare_bucket: str
    model_version: str | None
    latency_ms: float
    logged_to_db: bool


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(..., min_length=1)
    actual_fare: float = Field(..., ge=0, le=500)


class FeedbackResponse(BaseModel):
    prediction_id: str
    recorded: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    database_logging: bool
