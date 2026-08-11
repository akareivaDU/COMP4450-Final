"""Feature engineering shared by training, the API and the monitoring dashboard.

Keeping this in one module means the exact same transformation runs at training
time and at serving time, which avoids training/serving skew.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

FEATURE_COLUMNS = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "pu_location_id",
    "do_location_id",
]

TARGET_COLUMN = "fare_amount"

# Sanity bounds used to clean the raw TLC data and to clamp predictions.
MIN_FARE = 2.5
MAX_FARE = 200.0
MIN_DISTANCE = 0.1
MAX_DISTANCE = 100.0
MIN_PASSENGERS = 1
MAX_PASSENGERS = 6

# Buckets turn the continuous fare prediction into discrete classes so the
# monitoring dashboard can plot target drift.
FARE_BUCKETS = [
    (0.0, 10.0, "$0-10"),
    (10.0, 20.0, "$10-20"),
    (20.0, 30.0, "$20-30"),
    (30.0, 50.0, "$30-50"),
    (50.0, float("inf"), "$50+"),
]

FARE_BUCKET_LABELS = [label for _, _, label in FARE_BUCKETS]


def fare_bucket(fare: float) -> str:
    """Map a fare amount onto a discrete bucket label."""
    value = max(float(fare), 0.0)
    for low, high, label in FARE_BUCKETS:
        if low <= value < high:
            return label
    return FARE_BUCKET_LABELS[-1]


def time_features(pickup: datetime) -> dict[str, int]:
    """Derive calendar features from a single pickup timestamp."""
    return {
        "pickup_hour": int(pickup.hour),
        "pickup_dayofweek": int(pickup.weekday()),
        "is_weekend": int(pickup.weekday() >= 5),
    }


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are missing values or fall outside sane operating ranges."""
    required = [
        "fare_amount",
        "trip_distance",
        "passenger_count",
        "tpep_pickup_datetime",
        "PULocationID",
        "DOLocationID",
    ]
    cleaned = df.dropna(subset=required).copy()
    keep = (
        cleaned["fare_amount"].between(MIN_FARE, MAX_FARE)
        & cleaned["trip_distance"].between(MIN_DISTANCE, MAX_DISTANCE)
        & cleaned["passenger_count"].between(MIN_PASSENGERS, MAX_PASSENGERS)
    )
    return cleaned.loc[keep].reset_index(drop=True)


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn raw TLC trip records into a model matrix and target vector."""
    cleaned = clean_trips(df)
    pickup = pd.to_datetime(cleaned["tpep_pickup_datetime"])
    features = pd.DataFrame(
        {
            "trip_distance": cleaned["trip_distance"].astype(float),
            "passenger_count": cleaned["passenger_count"].astype(float),
            "pickup_hour": pickup.dt.hour.astype(int),
            "pickup_dayofweek": pickup.dt.weekday.astype(int),
            "is_weekend": (pickup.dt.weekday >= 5).astype(int),
            "pu_location_id": cleaned["PULocationID"].astype(int),
            "do_location_id": cleaned["DOLocationID"].astype(int),
        }
    )
    return features[FEATURE_COLUMNS], cleaned[TARGET_COLUMN].astype(float)


def build_features_row(
    trip_distance: float,
    passenger_count: int,
    pickup_datetime: datetime,
    pu_location_id: int,
    do_location_id: int,
) -> pd.DataFrame:
    """Build the single-row model matrix used to serve one prediction."""
    row = {
        "trip_distance": float(trip_distance),
        "passenger_count": float(passenger_count),
        **time_features(pickup_datetime),
        "pu_location_id": int(pu_location_id),
        "do_location_id": int(do_location_id),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
