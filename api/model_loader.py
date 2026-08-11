"""Loads the trained model.

By default the API pulls the model tagged "production" out of the Weights &
Biases model registry. A local joblib file is used as a fallback so the service
can still start (and the test suite can run) without network access to W&B.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

_BUNDLE: dict[str, Any] | None = None


def _model_source() -> str:
    return os.getenv("MODEL_SOURCE", "wandb").strip().lower()


def _local_path() -> Path:
    return Path(os.getenv("MODEL_LOCAL_PATH", "models/model.joblib"))


def _download_from_registry() -> Path:
    """Download the registered model artifact and return the path to the file."""
    import wandb

    ref = os.getenv("WANDB_MODEL_REF")
    if not ref:
        raise RuntimeError("WANDB_MODEL_REF is not set")
    artifact = wandb.Api().artifact(ref, type="model")
    directory = Path(artifact.download())
    logger.info("Downloaded model artifact %s (version %s)", ref, artifact.version)
    return directory / "model.joblib"


def load_model(force: bool = False) -> dict[str, Any]:
    """Return the cached model bundle, loading it on first use."""
    global _BUNDLE
    if _BUNDLE is not None and not force:
        return _BUNDLE

    path: Path | None = None
    if _model_source() == "wandb":
        try:
            path = _download_from_registry()
        except Exception as exc:  # noqa: BLE001 - fall back to the local copy
            logger.warning("Could not load model from W&B registry (%s); using local file", exc)
            path = None

    if path is None:
        path = _local_path()

    if not path.exists():
        raise FileNotFoundError(f"No model file found at {path}")

    _BUNDLE = joblib.load(path)
    return _BUNDLE


def try_load() -> bool:
    """Load the model but report failure instead of raising, for /health."""
    try:
        load_model()
        return True
    except Exception as exc:  # noqa: BLE001 - health check must not raise
        logger.warning("Model unavailable: %s", exc)
        return False


def is_loaded() -> bool:
    return _BUNDLE is not None


def model_version() -> str | None:
    if _BUNDLE is None:
        return None
    return _BUNDLE.get("model_version")
