
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.features import clean_trips

URL_TEMPLATE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"

COLUMNS = [
    "tpep_pickup_datetime",
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the training sample")
    parser.add_argument("--month", default="2026-01", help="Month to download, e.g. 2026-01")
    parser.add_argument("--file", default=None, help="Path to a parquet file already on disk")
    parser.add_argument("--rows", type=int, default=100_000, help="Rows to keep in the sample")
    parser.add_argument("--out", default="data/taxi_sample.csv")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.file:
        source = args.file
        print(f"Reading {source}")
    else:
        source = URL_TEMPLATE.format(month=args.month)
        print(f"Downloading {source}")

    frame = pd.read_parquet(source, columns=COLUMNS)
    print(f"Loaded {len(frame):,} rows")

    cleaned = clean_trips(frame)
    print(f"{len(cleaned):,} rows remain after cleaning")

    if len(cleaned) > args.rows:
        cleaned = cleaned.sample(n=args.rows, random_state=args.random_state)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    print(f"Wrote {len(cleaned):,} rows to {out_path}")


if __name__ == "__main__":
    main()
