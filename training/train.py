"""Train the taxi fare model and log the run to Weights & Biases.

Run this from the repository root:

    python -m training.train --max-iter 300 --learning-rate 0.1
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from common.features import FEATURE_COLUMNS, build_training_frame

DEFAULT_DATA = "data/taxi_sample.csv"
DEFAULT_MODEL_PATH = "models/model.joblib"


def git_commit() -> str:
    """Record the exact code version that produced this run."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - training should work outside a git checkout
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NYC taxi fare model")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=25)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--project", default="comp4450-taxi-fare")
    parser.add_argument("--entity", default=None, help="W&B entity (team or username)")
    parser.add_argument("--model-name", default="taxi-fare-model")
    parser.add_argument(
        "--registry-path",
        default="wandb-registry-model/taxi-fare-model",
        help="Registry collection to link the model into; pass an empty string to skip",
    )
    parser.add_argument("--alias", default="production")
    parser.add_argument("--no-wandb", action="store_true", help="Train without logging to W&B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = pd.read_csv(args.data)
    features, target = build_training_frame(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=args.test_size, random_state=args.random_state
    )

    config = {
        "learning_rate": args.learning_rate,
        "max_iter": args.max_iter,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "model_type": "HistGradientBoostingRegressor",
        "feature_columns": FEATURE_COLUMNS,
        "git_commit": git_commit(),
        "data_file": args.data,
        "n_rows_raw": int(len(frame)),
        "n_rows_clean": int(len(features)),
    }

    run = None
    wandb = None
    if not args.no_wandb:
        import wandb as wandb_module

        wandb = wandb_module
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            job_type="train",
            config=config,
        )

    model = HistGradientBoostingRegressor(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    errors = np.abs(predictions - y_test.to_numpy())
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "within_2_dollars": float(np.mean(errors <= 2.0)),
        "within_5_dollars": float(np.mean(errors <= 5.0)),
    }

    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    model_version = run.id if run is not None else f"local-{trained_at}"
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "model_version": model_version,
        "trained_at": trained_at,
        "git_commit": config["git_commit"],
        "metrics": metrics,
    }

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    if run is not None and wandb is not None:
        run.log(metrics)
        run.summary.update(metrics)

        dataset_artifact = wandb.Artifact(
            "taxi-trips",
            type="dataset",
            metadata={"rows": int(len(frame)), "source": args.data},
        )
        dataset_artifact.add_file(args.data)
        run.log_artifact(dataset_artifact)

        model_artifact = wandb.Artifact(
            args.model_name,
            type="model",
            metadata={**metrics, "git_commit": config["git_commit"]},
        )
        model_artifact.add_file(str(model_path))
        logged_artifact = run.log_artifact(model_artifact)
        logged_artifact.wait()

        if args.registry_path:
            try:
                run.link_artifact(logged_artifact, args.registry_path, aliases=[args.alias])
                print(f"Linked model to {args.registry_path}:{args.alias}")
            except Exception as exc:  # noqa: BLE001 - registry setup issues shouldn't lose the run
                print(f"Could not link to registry '{args.registry_path}': {exc}")

        run.finish()

    print(json.dumps({"model_version": model_version, **metrics}, indent=2))


if __name__ == "__main__":
    main()
