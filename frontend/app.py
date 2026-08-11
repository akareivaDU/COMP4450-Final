"""User-facing Streamlit app: send a trip to the API and see the predicted fare."""

from __future__ import annotations

import os
from datetime import datetime
from datetime import time as dtime

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 15

# A handful of well-known TLC zone ids so the form is usable without the full
# lookup table. Any id from 1 to 265 is accepted.
ZONES = {
    "JFK Airport (132)": 132,
    "LaGuardia Airport (138)": 138,
    "Newark Airport (1)": 1,
    "Midtown Center (161)": 161,
    "Times Sq / Theatre District (230)": 230,
    "Penn Station / Madison Sq West (186)": 186,
    "Upper East Side North (236)": 236,
    "Upper East Side South (237)": 237,
    "Lincoln Square East (142)": 142,
    "Union Sq (234)": 234,
    "East Village (79)": 79,
    "West Village (249)": 249,
    "Central Park (43)": 43,
    "Garment District (100)": 100,
}

st.set_page_config(page_title="NYC Taxi Fare Estimator", page_icon=None, layout="centered")
st.title("NYC Taxi Fare Estimator")
st.caption(f"Backend: {API_URL}")


def api_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


health = api_health()
if health is None:
    st.error("Cannot reach the prediction API. Check that the backend is running.")
elif not health.get("model_loaded"):
    st.warning("The API is up but no model is loaded.")
else:
    st.success(f"API healthy. Model version: {health.get('model_version')}")

st.subheader("Trip details")

col1, col2 = st.columns(2)
with col1:
    pickup_zone = st.selectbox("Pickup zone", list(ZONES), index=3)
    trip_distance = st.number_input("Trip distance (miles)", 0.1, 100.0, 3.2, step=0.1)
    pickup_date = st.date_input("Pickup date", datetime.now().date())
with col2:
    dropoff_zone = st.selectbox("Dropoff zone", list(ZONES), index=0)
    passenger_count = st.number_input("Passengers", 1, 6, 1, step=1)
    pickup_time = st.time_input("Pickup time", dtime(18, 30))

if st.button("Predict fare", type="primary"):
    payload = {
        "trip_distance": float(trip_distance),
        "passenger_count": int(passenger_count),
        "pu_location_id": ZONES[pickup_zone],
        "do_location_id": ZONES[dropoff_zone],
        "pickup_datetime": datetime.combine(pickup_date, pickup_time).isoformat(),
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        st.session_state["prediction"] = response.json()
    except requests.RequestException as exc:
        st.error(f"Prediction failed: {exc}")

prediction = st.session_state.get("prediction")
if prediction:
    st.subheader("Prediction")
    left, middle, right = st.columns(3)
    left.metric("Estimated fare", f"${prediction['predicted_fare']:.2f}")
    middle.metric("Fare bucket", prediction["fare_bucket"])
    right.metric("Latency", f"{prediction['latency_ms']:.1f} ms")
    st.caption(f"Prediction id: {prediction['prediction_id']}")
    if not prediction.get("logged_to_db"):
        st.warning("This prediction was not written to the database.")

    st.subheader("Was this right?")
    st.write("Enter the fare you actually paid. This feeds the live accuracy metrics.")
    actual_fare = st.number_input("Actual fare paid ($)", 0.0, 500.0, 0.0, step=0.5)
    if st.button("Submit feedback"):
        try:
            response = requests.post(
                f"{API_URL}/feedback",
                json={
                    "prediction_id": prediction["prediction_id"],
                    "actual_fare": float(actual_fare),
                },
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            st.success("Thanks, your feedback was recorded.")
        except requests.RequestException as exc:
            st.error(f"Could not record feedback: {exc}")
