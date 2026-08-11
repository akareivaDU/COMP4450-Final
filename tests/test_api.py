"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def valid_payload(**overrides) -> dict:
    payload = {
        "trip_distance": 4.5,
        "passenger_count": 2,
        "pu_location_id": 132,
        "do_location_id": 230,
        "pickup_datetime": "2024-01-06T18:30:00",
    }
    payload.update(overrides)
    return payload


def test_health_reports_a_loaded_model(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-stub"


def test_root_returns_service_information(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "taxi-fare-api"


def test_predict_returns_a_positive_fare(client: TestClient):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_fare"] > 0
    assert body["prediction_id"]
    assert body["latency_ms"] >= 0
    assert body["fare_bucket"] in {"$0-10", "$10-20", "$20-30", "$30-50", "$50+"}
    assert body["model_version"] == "test-stub"


def test_predict_works_without_a_pickup_time(client: TestClient):
    payload = valid_payload()
    payload.pop("pickup_datetime")
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_longer_trips_cost_more(client: TestClient):
    short = client.post("/predict", json=valid_payload(trip_distance=1.0)).json()
    long = client.post("/predict", json=valid_payload(trip_distance=18.0)).json()
    assert long["predicted_fare"] > short["predicted_fare"]


@pytest.mark.parametrize(
    "override",
    [
        {"trip_distance": 0},
        {"trip_distance": -5},
        {"passenger_count": 0},
        {"passenger_count": 12},
        {"pu_location_id": 0},
        {"do_location_id": 900},
    ],
)
def test_predict_rejects_invalid_input(client: TestClient, override: dict):
    response = client.post("/predict", json=valid_payload(**override))
    assert response.status_code == 422


def test_predict_requires_required_fields(client: TestClient):
    response = client.post("/predict", json={"passenger_count": 1})
    assert response.status_code == 422


def test_feedback_endpoint_accepts_a_submission(client: TestClient):
    prediction = client.post("/predict", json=valid_payload()).json()
    response = client.post(
        "/feedback",
        json={"prediction_id": prediction["prediction_id"], "actual_fare": 21.75},
    )
    assert response.status_code == 200
    # Database logging is disabled in tests, so nothing is persisted.
    assert response.json()["recorded"] is False


def test_feedback_rejects_an_impossible_fare(client: TestClient):
    response = client.post(
        "/feedback", json={"prediction_id": "abc", "actual_fare": -10}
    )
    assert response.status_code == 422
