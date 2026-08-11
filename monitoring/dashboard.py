"""Model monitoring dashboard.

Runs on its own EC2 instance and talks to the API only indirectly: it reads the
prediction logs the API wrote to DynamoDB.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from common import db
from common.features import FARE_BUCKET_LABELS, fare_bucket

MAE_ALERT_THRESHOLD = float(os.getenv("MAE_ALERT_THRESHOLD", "5.0"))
MIN_FEEDBACK_FOR_ALERT = int(os.getenv("MIN_FEEDBACK_FOR_ALERT", "10"))

st.set_page_config(page_title="Taxi Fare Model Monitoring", layout="wide")
st.title("Taxi Fare Model Monitoring")
st.caption(f"DynamoDB table: {db.table_name()} ({db.region()})")


@st.cache_data(ttl=30, show_spinner="Loading prediction logs...")
def load_logs(limit: int = 5000) -> pd.DataFrame:
    items = db.scan_predictions(limit=limit)
    if not items:
        return pd.DataFrame()

    frame = pd.DataFrame(items)
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    for column in ["predicted_fare", "latency_ms", "trip_distance", "actual_fare"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = pd.NA

    # Recompute buckets so the chart is consistent even for older records.
    frame["fare_bucket"] = frame["predicted_fare"].apply(
        lambda value: fare_bucket(value) if pd.notna(value) else None
    )
    return frame.dropna(subset=["created_at"]).sort_values("created_at")


if st.button("Refresh data"):
    load_logs.clear()

logs = load_logs()

if logs.empty:
    st.info("No predictions logged yet. Make a prediction in the frontend app first.")
    st.stop()

# Missing feedback arrives as NaN, not None, once it goes through pandas.
# Every accuracy calculation below has to filter on notna() first.
labeled = logs[logs["actual_fare"].notna()].copy()

st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictions logged", f"{len(logs):,}")
c2.metric("Average latency", f"{logs['latency_ms'].mean():.1f} ms")
c3.metric("Average predicted fare", f"${logs['predicted_fare'].mean():.2f}")
c4.metric("Feedback received", f"{len(labeled):,}")

st.subheader("Live accuracy")
if labeled.empty:
    st.info("No user feedback yet, so live accuracy cannot be computed.")
else:
    errors = (labeled["predicted_fare"] - labeled["actual_fare"]).abs()
    mae = float(errors.mean())
    rmse = float(((labeled["predicted_fare"] - labeled["actual_fare"]) ** 2).mean() ** 0.5)
    within_two = float((errors <= 2.0).mean() * 100)

    a1, a2, a3 = st.columns(3)
    a1.metric("Live MAE", f"${mae:.2f}")
    a2.metric("Live RMSE", f"${rmse:.2f}")
    a3.metric("Within $2", f"{within_two:.1f}%")

    if len(labeled) >= MIN_FEEDBACK_FOR_ALERT and mae > MAE_ALERT_THRESHOLD:
        st.error(
            f"Model performance alert: live MAE is ${mae:.2f}, "
            f"above the ${MAE_ALERT_THRESHOLD:.2f} threshold."
        )
    else:
        st.success("Live error is within the acceptable threshold.")

    st.scatter_chart(
        labeled.rename(columns={"actual_fare": "Actual fare", "predicted_fare": "Predicted fare"}),
        x="Actual fare",
        y="Predicted fare",
        height=320,
    )

st.subheader("Prediction latency over time")
latency = (
    logs.set_index("created_at")["latency_ms"].resample("5min").mean().dropna().to_frame("Mean ms")
)
st.line_chart(latency, height=260)

left, right = st.columns(2)

with left:
    st.subheader("Predicted fare distribution")
    counts = (
        logs["fare_bucket"]
        .value_counts()
        .reindex(FARE_BUCKET_LABELS, fill_value=0)
        .to_frame("Predictions")
    )
    st.bar_chart(counts, height=280)

with right:
    st.subheader("Input drift: trip distance")
    bins = pd.cut(logs["trip_distance"].dropna(), bins=[0, 1, 2, 5, 10, 20, 1000])
    distance_counts = bins.value_counts().sort_index()
    distance_counts.index = distance_counts.index.astype(str)
    st.bar_chart(distance_counts.to_frame("Trips"), height=280)

st.subheader("Target drift: fare bucket share over time")
hourly = (
    logs.set_index("created_at")
    .groupby([pd.Grouper(freq="1h"), "fare_bucket"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=FARE_BUCKET_LABELS, fill_value=0)
)
if len(hourly) > 1:
    share = hourly.div(hourly.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0)
    st.area_chart(share, height=300)
else:
    st.info("Need predictions across more than one hour to show drift over time.")

st.subheader("Submit feedback")
st.write("Record the fare that was actually paid for a logged prediction.")
pending = logs[logs["actual_fare"].isna()].sort_values("created_at", ascending=False)
if pending.empty:
    st.info("Every logged prediction already has feedback.")
else:
    options = pending["prediction_id"].head(50).tolist()
    selected = st.selectbox("Prediction id", options)
    entered_fare = st.number_input("Actual fare paid ($)", 0.0, 500.0, 0.0, step=0.5)
    if st.button("Record feedback"):
        try:
            db.record_feedback(selected, float(entered_fare))
            load_logs.clear()
            st.success("Feedback recorded.")
        except db.PredictionNotFound:
            st.error("That prediction id no longer exists.")
        except Exception as exc:  # noqa: BLE001 - show the AWS error to the operator
            st.error(f"Could not record feedback: {exc}")

with st.expander("Recent prediction logs"):
    st.dataframe(logs.sort_values("created_at", ascending=False).head(100), width="stretch")
