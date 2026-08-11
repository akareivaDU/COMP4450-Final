"""Create the DynamoDB table used for prediction logs.

    python scripts/create_dynamodb_table.py --region us-east-1
"""

from __future__ import annotations

import argparse

import boto3
from botocore.exceptions import ClientError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the prediction log table")
    parser.add_argument("--table", default="taxi-fare-predictions")
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = boto3.client("dynamodb", region_name=args.region)

    try:
        client.create_table(
            TableName=args.table,
            AttributeDefinitions=[{"AttributeName": "prediction_id", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "prediction_id", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ResourceInUseException":
            print(f"Table '{args.table}' already exists in {args.region}")
            return
        raise

    print(f"Creating table '{args.table}' in {args.region}...")
    client.get_waiter("table_exists").wait(TableName=args.table)
    print("Table is active.")


if __name__ == "__main__":
    main()
