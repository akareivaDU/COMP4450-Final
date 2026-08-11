"""DynamoDB access layer shared by the API and the monitoring dashboard.

The API writes prediction logs here and the dashboard reads them back. This is
the only channel between the two services, which is why they can live on
separate EC2 instances.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import boto3
from botocore.exceptions import ClientError

_TABLE = None


class PredictionNotFound(Exception):
    """Raised when feedback is submitted for a prediction id that does not exist."""


def table_name() -> str:
    return os.getenv("DYNAMODB_TABLE", "taxi-fare-predictions")


def region() -> str:
    return os.getenv("AWS_REGION", "us-east-1")


def enabled() -> bool:
    """Database logging can be switched off for local runs and CI."""
    return os.getenv("LOG_TO_DB", "true").strip().lower() in {"1", "true", "yes"}


def get_table():
    global _TABLE
    if _TABLE is None:
        _TABLE = boto3.resource("dynamodb", region_name=region()).Table(table_name())
    return _TABLE


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def to_decimal(value: Any) -> Decimal:
    """DynamoDB rejects Python floats, so every number is stored as a Decimal."""
    return Decimal(str(round(float(value), 4)))


def _encode(item: dict[str, Any]) -> dict[str, Any]:
    encoded: dict[str, Any] = {}
    for key, value in item.items():
        if value is None:
            continue
        if isinstance(value, bool):
            encoded[key] = value
        elif isinstance(value, (int, float, Decimal)):
            encoded[key] = to_decimal(value)
        else:
            encoded[key] = str(value)
    return encoded


def decode(item: dict[str, Any]) -> dict[str, Any]:
    """Convert DynamoDB Decimals back into plain floats for pandas."""
    return {k: (float(v) if isinstance(v, Decimal) else v) for k, v in item.items()}


def log_prediction(item: dict[str, Any]) -> bool:
    """Write one prediction log record. Returns False when logging is disabled."""
    if not enabled():
        return False
    get_table().put_item(Item=_encode(item))
    return True


def record_feedback(prediction_id: str, actual_fare: float) -> bool:
    """Attach the real fare reported by a user to an existing prediction."""
    if not enabled():
        return False
    try:
        get_table().update_item(
            Key={"prediction_id": prediction_id},
            UpdateExpression="SET actual_fare = :fare, feedback_at = :ts",
            ExpressionAttributeValues={
                ":fare": to_decimal(actual_fare),
                ":ts": utc_now_iso(),
            },
            ConditionExpression="attribute_exists(prediction_id)",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            raise PredictionNotFound(prediction_id) from exc
        raise
    return True


def scan_predictions(limit: int = 5000) -> list[dict[str, Any]]:
    """Read prediction logs back out of DynamoDB.

    A full scan is fine at course-project volumes. For a real production table
    you would add a global secondary index on the timestamp and query a window.
    """
    table = get_table()
    items: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {}
    while True:
        response = table.scan(**kwargs)
        items.extend(decode(item) for item in response.get("Items", []))
        key = response.get("LastEvaluatedKey")
        if key is None or len(items) >= limit:
            break
        kwargs["ExclusiveStartKey"] = key
    return items[:limit]
