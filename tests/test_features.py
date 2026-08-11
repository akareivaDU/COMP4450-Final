"""Unit tests for the shared preprocessing logic."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from common.features import (
    FEATURE_COLUMNS,
    build_features_row,
    build_training_frame,
    clean_trips,
    fare_bucket,
    time_features,
)


def sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2024-01-06 18:30:00",  # Saturday
                "2024-01-08 09:15:00",  # Monday
                "2024-01-08 09:20:00",
                "2024-01-08 09:25:00",
            ],
            "trip_distance": [3.0, 1.5, 0.0, 4.0],  # third row is too short
            "passenger_count": [1, 2, 1, 9],  # fourth row has too many passengers
            "PULocationID": [132, 161, 100, 138],
            "DOLocationID": [230, 236, 142, 79],
            "fare_amount": [18.5, 9.0, 5.0, 25.0],
        }
    )


def test_time_features_for_a_weekend_evening():
    features = time_features(datetime(2024, 1, 6, 18, 30))
    assert features == {"pickup_hour": 18, "pickup_dayofweek": 5, "is_weekend": 1}


def test_time_features_for_a_weekday_morning():
    features = time_features(datetime(2024, 1, 8, 9, 15))
    assert features["is_weekend"] == 0
    assert features["pickup_dayofweek"] == 0


def test_build_features_row_column_order_matches_training():
    row = build_features_row(
        trip_distance=3.2,
        passenger_count=2,
        pickup_datetime=datetime(2024, 1, 6, 18, 30),
        pu_location_id=132,
        do_location_id=230,
    )
    assert list(row.columns) == FEATURE_COLUMNS
    assert len(row) == 1
    assert row.loc[0, "trip_distance"] == 3.2
    assert row.loc[0, "pu_location_id"] == 132


def test_clean_trips_removes_out_of_range_rows():
    cleaned = clean_trips(sample_raw_frame())
    assert len(cleaned) == 2
    assert cleaned["trip_distance"].min() > 0
    assert cleaned["passenger_count"].max() <= 6


def test_clean_trips_drops_missing_values():
    frame = sample_raw_frame()
    frame.loc[0, "fare_amount"] = None
    cleaned = clean_trips(frame)
    assert len(cleaned) == 1


def test_build_training_frame_shapes_and_types():
    features, target = build_training_frame(sample_raw_frame())
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == len(target) == 2
    assert features["is_weekend"].tolist() == [1, 0]


def test_fare_bucket_boundaries():
    assert fare_bucket(0) == "$0-10"
    assert fare_bucket(9.99) == "$0-10"
    assert fare_bucket(10) == "$10-20"
    assert fare_bucket(29.5) == "$20-30"
    assert fare_bucket(50) == "$50+"
    assert fare_bucket(999) == "$50+"


def test_fare_bucket_handles_negative_values():
    assert fare_bucket(-3.0) == "$0-10"
