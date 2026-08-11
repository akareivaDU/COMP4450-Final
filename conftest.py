"""Test configuration.

This runs before any test module is imported, so it is where the environment is
pointed at a small stub model and database writes are switched off.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from common.features import FEATURE_COLUMNS

_TMP_DIR = Path(tempfile.mkdtemp(prefix="taxi-test-"))
_MODEL_PATH = _TMP_DIR / "model.joblib"


def _build_stub_model() -> None:
    """Train a tiny model on synthetic data so the API has something to serve."""
    rng = np.random.default_rng(0)
    size = 500
    frame = pd.DataFrame(
        {
            "trip_distance": rng.uniform(0.5, 20.0, size),
            "passenger_count": rng.integers(1, 7, size).astype(float),
            "pickup_hour": rng.integers(0, 24, size),
            "pickup_dayofweek": rng.integers(0, 7, size),
            "is_weekend": rng.integers(0, 2, size),
            "pu_location_id": rng.integers(1, 266, size),
            "do_location_id": rng.integers(1, 266, size),
        }
    )[FEATURE_COLUMNS]
    target = 3.0 + 2.6 * frame["trip_distance"] + rng.normal(0, 0.8, size)

    model = HistGradientBoostingRegressor(max_iter=30, random_state=0).fit(frame, target)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "model_version": "test-stub",
            "trained_at": "1970-01-01T00:00:00+00:00",
            "metrics": {},
        },
        _MODEL_PATH,
    )


os.environ["LOG_TO_DB"] = "false"
os.environ["MODEL_SOURCE"] = "local"
os.environ["MODEL_LOCAL_PATH"] = str(_MODEL_PATH)

_build_stub_model()
