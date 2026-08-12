# NYC Taxi Fare Prediction - Production MLOps System

An end-to-end machine learning service that predicts yellow taxi fares in New York City. The system covers experiment tracking, a model registry, a prediction API, a cloud database, a user frontend, a live monitoring dashboard, automated testing, and CI/CD.

## Architecture

```
                Weights & Biases
              (experiments + model registry)
                        |
                        | pulls "production" model at startup
                        v
  EC2 instance 1                           EC2 instance 2
  +----------------------------+           +--------------------------+
  |  Streamlit frontend :8501  |           |  Streamlit monitoring    |
  |            |               |           |  dashboard :8501         |
  |            v HTTP          |           +--------------------------+
  |  FastAPI backend :8000     |                       ^
  +----------------------------+                       |
                |                                      |
                | writes prediction logs               | reads logs
                v                                      |
              +------------------------------------------+
              |  Amazon DynamoDB: taxi-fare-predictions  |
              +------------------------------------------+
```

The two EC2 instances never talk to each other. All data is exchanged through DynamoDB.

## Components

| Component | Path | Description |
|---|---|---|
| Shared preprocessing | `common/features.py` | Feature engineering used by training, the API, and the dashboard |
| Database layer | `common/db.py` | DynamoDB reads and writes |
| Training | `training/train.py` | Trains the model and logs the run to Weights & Biases |
| Prediction API | `api/` | FastAPI service with `/predict`, `/feedback`, `/health` |
| User frontend | `frontend/app.py` | Streamlit form that calls the API |
| Monitoring dashboard | `monitoring/dashboard.py` | Streamlit dashboard reading DynamoDB |
| Tests | `tests/` | Unit tests and API integration tests |
| CI | `.github/workflows/ci.yml` | Ruff lint plus pytest on every pull request |

## Model

A `HistGradientBoostingRegressor` from scikit-learn predicts `fare_amount` from seven features:

- `trip_distance`
- `passenger_count`
- `pickup_hour`
- `pickup_dayofweek`
- `is_weekend`
- `pu_location_id`
- `do_location_id`

Because fare prediction is a regression problem, the monitoring metrics are adapted accordingly. Live accuracy is measured with mean absolute error, root mean squared error, and the share of predictions within two dollars of the actual fare. Predicted fares are also mapped into five discrete buckets so that target drift can be plotted as a class distribution.

## Local setup

Requires Python 3.12 or newer, Docker, an AWS account, and a Weights & Biases account.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r training/requirements.txt
```

Copy the environment template and fill it in:

```bash
cp .env.example .env
```

### 1. Download the data

```bash
python -m data.download_data --month 2026-01 --rows 100000
```

If you already have the parquet file on disk, point at it instead of downloading:

```bash
python -m data.download_data --file yellow_tripdata_2026-01.parquet
```

Either way this applies the cleaning rules in `common/features.py` and writes a sample to `data/taxi_sample.csv`. January 2026 has 3.72M raw trips; about 68 percent survive cleaning, and 100,000 of those are sampled into a 4 MB CSV that is small enough to commit.

### 2. Create the DynamoDB table

```bash
export AWS_REGION=us-east-1
python scripts/create_dynamodb_table.py --region us-east-1
```

The table uses `prediction_id` as its partition key and on-demand billing.

### 3. Train and register the model

```bash
wandb login

python -m training.train --learning-rate 0.1 --max-iter 300 --max-depth 8
python -m training.train --learning-rate 0.05 --max-iter 500 --max-depth 6
python -m training.train --learning-rate 0.2 --max-iter 200 --max-depth 10
```

Each run logs hyperparameters, the git commit hash, the dataset artifact, and the metrics (MAE, RMSE, R2, and the within-two-dollars rate) to Weights & Biases. The trained model is saved as a versioned model artifact and linked into the model registry with the alias `production`.

Compare the runs in the W&B dashboard and re-link the best one if a later run wins:

```bash
python -m training.train --learning-rate 0.05 --max-iter 500 --alias production
```

If your W&B account uses the older registry layout, pass the legacy path instead:

```bash
python -m training.train --registry-path "<entity>/model-registry/Taxi Fare Model"
```

### 4. Run everything locally

```bash
docker compose up --build
DASHBOARD_PORT=8502 docker compose -f docker-compose.monitoring.yml up --build
```

- API and docs: http://localhost:8000/docs
- User frontend: http://localhost:8501
- Monitoring dashboard: http://localhost:8502

To run without W&B, set `MODEL_SOURCE=local` in `.env` and the API will read `models/model.joblib` instead.

## API reference

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "a1b2c3d4",
  "database_logging": true
}
```

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trip_distance": 8.4,
    "passenger_count": 2,
    "pu_location_id": 132,
    "do_location_id": 230,
    "pickup_datetime": "2024-01-06T18:30:00"
  }'
```

```json
{
  "prediction_id": "0f2c1e5a-6b0d-4d3f-9a2e-1c7b8d4e5f60",
  "predicted_fare": 42.18,
  "fare_bucket": "$30-50",
  "model_version": "a1b2c3d4",
  "latency_ms": 3.41,
  "logged_to_db": true
}
```

`pickup_datetime` is optional and defaults to the current UTC time. `pu_location_id` and `do_location_id` are TLC taxi zone ids between 1 and 265; 132 is JFK, 138 is LaGuardia, and 230 is Times Square.

### POST /feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "0f2c1e5a-6b0d-4d3f-9a2e-1c7b8d4e5f60",
    "actual_fare": 44.50
  }'
```

```json
{
  "prediction_id": "0f2c1e5a-6b0d-4d3f-9a2e-1c7b8d4e5f60",
  "recorded": true,
  "message": "Feedback recorded"
}
```

Feedback attaches the real fare to a stored prediction, which is what makes live accuracy measurable.

## Monitoring dashboard

The dashboard reads the prediction log table directly and shows:

- Prediction latency over time, averaged in five minute windows
- Distribution of predicted fare buckets, which is the target distribution
- Fare bucket share per hour, which shows target drift
- Trip distance histogram, which shows input drift
- Live MAE, RMSE, and within-two-dollars rate from user feedback
- A predicted versus actual scatter plot
- A red alert banner when live MAE exceeds `MAE_ALERT_THRESHOLD` (default 5.0) with at least `MIN_FEEDBACK_FOR_ALERT` feedback records
- A form for recording feedback directly from the dashboard

## Testing and CI

```bash
ruff check .
pytest -v
```

Unit tests in `tests/test_features.py` cover the preprocessing logic: calendar feature extraction, row cleaning, feature column ordering, and fare bucketing. Integration tests in `tests/test_api.py` run the FastAPI app through `TestClient` and cover the health check, successful predictions, monotonic behavior on distance, request validation, and the feedback endpoint.

Tests do not require AWS or W&B. `conftest.py` sets `LOG_TO_DB=false` and points the API at a small stub model trained on synthetic data.

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on every pull request to `main` and on pushes to `main`. It installs dependencies, runs `ruff check .`, then runs `pytest -v`.

To block merges when checks fail, go to Settings, Branches, Add branch protection rule for `main`, enable "Require status checks to pass before merging", and select the `lint-and-test` check.

## AWS deployment

### DynamoDB

Create the table once with `scripts/create_dynamodb_table.py`, or through the AWS console with partition key `prediction_id` of type String and on-demand capacity.

### IAM role

Create an IAM role for EC2 with a policy granting `dynamodb:PutItem`, `dynamodb:UpdateItem`, `dynamodb:GetItem`, and `dynamodb:Scan` on the table ARN. Attach it to both instances so no AWS keys need to live in environment files.

### EC2 instance 1: API and frontend

Launch an Amazon Linux 2023 t3.small instance, attach the IAM role, and open ports 22, 8000, and 8501 to your IP in the security group.

```bash
sudo dnf update -y
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

cp .env.example .env
nano .env    # set WANDB_API_KEY, WANDB_MODEL_REF, AWS_REGION, DYNAMODB_TABLE

docker compose up -d --build
docker compose ps
curl http://localhost:8000/health
```

### EC2 instance 2: monitoring dashboard

Launch a second instance the same way, attach the same IAM role, and open ports 22 and 8501.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

cp .env.example .env
nano .env    # set AWS_REGION and DYNAMODB_TABLE

docker compose -f docker-compose.monitoring.yml up -d --build
```

The dashboard is then reachable at `http://<instance-2-public-ip>:8501` and the frontend at `http://<instance-1-public-ip>:8501`.

## Environment variables

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `MODEL_SOURCE` | API | `wandb` | `wandb` pulls from the registry, `local` reads a file |
| `WANDB_API_KEY` | API | none | Authenticates the registry download |
| `WANDB_MODEL_REF` | API | none | e.g. `org/wandb-registry-model/taxi-fare-model:production` |
| `MODEL_LOCAL_PATH` | API | `models/model.joblib` | Fallback model file |
| `AWS_REGION` | API, dashboard | `us-east-1` | DynamoDB region |
| `DYNAMODB_TABLE` | API, dashboard | `taxi-fare-predictions` | Prediction log table |
| `LOG_TO_DB` | API | `true` | Set to `false` to disable database writes |
| `API_URL` | frontend | `http://api:8000` | Backend address |
| `MAE_ALERT_THRESHOLD` | dashboard | `5.0` | MAE above this triggers the alert banner |
| `MIN_FEEDBACK_FOR_ALERT` | dashboard | `10` | Minimum feedback count before alerting |

## Notes

Keep the scikit-learn version used for training close to the version in `api/requirements.txt`. Loading a model that was pickled by a different minor version produces a version warning and can change behavior.

The dashboard scans the whole DynamoDB table. That is fine at course-project volumes. A production system would add a global secondary index on the timestamp and query a time window instead.

`models/model.joblib` is gitignored. Either let the API pull the model from the W&B registry, or run the training script on the instance to regenerate the local fallback.

## Project structure

```
.
├── api/                        FastAPI prediction service
│   ├── Dockerfile
│   ├── main.py
│   ├── model_loader.py
│   ├── requirements.txt
│   └── schemas.py
├── common/                     Code shared across services
│   ├── db.py
│   └── features.py
├── data/
│   └── download_data.py
├── frontend/                   User-facing Streamlit app
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── monitoring/                 Monitoring dashboard
│   ├── Dockerfile
│   ├── dashboard.py
│   └── requirements.txt
├── scripts/
│   └── create_dynamodb_table.py
├── tests/
│   ├── test_api.py
│   └── test_features.py
├── training/
│   ├── requirements.txt
│   └── train.py
├── .github/workflows/ci.yml
├── conftest.py
├── docker-compose.yml
├── docker-compose.monitoring.yml
├── requirements-dev.txt
└── ruff.toml
```

