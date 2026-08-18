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
git clone https://github.com/akareivaDU/COMP4450-Final.git
cd COMP4450-Final

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
export AWS_REGION=us-east-2
python scripts/create_dynamodb_table.py --region us-east-2
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

The registry reference must not be prefixed with the entity name. `akareiva4-denver-university/wandb-registry-model/...` fails with "Unable to find organization for entity"; the bare collection path works because W&B resolves the organization from the API key.

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

Launch an Amazon Linux 2023 t3.small instance, attach the IAM role, and open ports 22, 8000, and 8501 to your IP in the security group. Use t3.small rather than t2.micro; 1 GB of memory is not enough to build the images.

Install Docker and git:

```bash
sudo dnf update -y
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker
```

Amazon Linux 2023 ships only the Docker engine, so the Compose and buildx plugins have to be installed separately. Without them `docker compose up --build` fails with "compose is not a docker command" and then "compose build requires buildx".

```bash
sudo mkdir -p /usr/libexec/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/libexec/docker/cli-plugins/docker-compose
sudo curl -SL https://github.com/docker/buildx/releases/download/v0.36.1/buildx-v0.36.1.linux-amd64 \
  -o /usr/libexec/docker/cli-plugins/docker-buildx
sudo chmod +x /usr/libexec/docker/cli-plugins/docker-compose \
  /usr/libexec/docker/cli-plugins/docker-buildx

docker compose version
docker buildx version
```

Then clone the repository and start the services:

```bash
git clone https://github.com/akareivaDU/COMP4450-Final.git
cd COMP4450-Final

cat > .env <<'ENV'
MODEL_SOURCE=wandb
WANDB_API_KEY=your-key-here
WANDB_MODEL_REF=wandb-registry-model/taxi-fare-model:production
LOG_TO_DB=true
AWS_REGION=us-east-2
DYNAMODB_TABLE=taxi-fare-predictions
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
API_URL=http://api:8000
ENV
nano .env    # replace your-key-here with the real W&B key

docker compose up -d --build
docker compose ps
curl http://localhost:8000/health
```

Leave `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` empty. boto3 picks up credentials from the attached IAM instance role, so no keys need to be stored on the server.

### EC2 instance 2: monitoring dashboard

Launch a second Amazon Linux 2023 t3.small instance, attach the same IAM role, and open ports 22 and 8501. This instance does not need port 8000; it never talks to the API directly.

Run the same Docker, git, and plugin installation steps as instance 1, then:

```bash
git clone https://github.com/akareivaDU/COMP4450-Final.git
cd COMP4450-Final

cat > .env <<'ENV'
AWS_REGION=us-east-2
DYNAMODB_TABLE=taxi-fare-predictions
MAE_ALERT_THRESHOLD=5.0
MIN_FEEDBACK_FOR_ALERT=10
DASHBOARD_PORT=8501
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
ENV

docker compose -f docker-compose.monitoring.yml up -d --build
docker compose -f docker-compose.monitoring.yml ps
```

No W&B credentials are needed here. The dashboard only reads prediction logs from DynamoDB and never loads the model.

The dashboard is then reachable at `http://<instance-2-public-ip>:8501` and the frontend at `http://<instance-1-public-ip>:8501`.

### Verifying the deployment

Make a prediction on instance 1, either through the frontend or directly against the API:

```bash
curl -X POST http://<instance-1-public-ip>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_distance": 8.4, "passenger_count": 2, "pu_location_id": 132, "do_location_id": 230}'
```

A response with `"logged_to_db": true` confirms the IAM instance role reached DynamoDB. The prediction should then appear on the monitoring dashboard on instance 2 after clicking Refresh data, which demonstrates that the two instances exchange data only through the database.

## Environment variables

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `MODEL_SOURCE` | API | `wandb` | `wandb` pulls from the registry, `local` reads a file |
| `WANDB_API_KEY` | API | none | Authenticates the registry download |
| `WANDB_MODEL_REF` | API | none | e.g. `wandb-registry-model/taxi-fare-model:production` |
| `MODEL_LOCAL_PATH` | API | `models/model.joblib` | Fallback model file |
| `AWS_REGION` | API, dashboard | `us-east-1` | DynamoDB region; this project uses `us-east-2` |
| `DYNAMODB_TABLE` | API, dashboard | `taxi-fare-predictions` | Prediction log table |
| `LOG_TO_DB` | API | `true` | Set to `false` to disable database writes |
| `API_URL` | frontend | `http://api:8000` | Backend address |
| `MAE_ALERT_THRESHOLD` | dashboard | `5.0` | MAE above this triggers the alert banner |
| `MIN_FEEDBACK_FOR_ALERT` | dashboard | `10` | Minimum feedback count before alerting |

## Deployment notes

Amazon Linux 2023 installs only the Docker engine. The Compose and buildx CLI plugins must be added manually before `docker compose up --build` will work; see the deployment section above.

Use t3.small or larger for the EC2 instances. A t2.micro has 1 GB of memory and gets out-of-memory killed while pip installs scikit-learn during the image build.

The monitoring Dockerfile sets `PYTHONPATH=/app`. Streamlit puts the script's own directory on `sys.path` rather than the working directory, so without it `from common import db` fails with `ModuleNotFoundError` inside the container. The API is unaffected because uvicorn adds the working directory itself.

Scripts that import from `common/` must be run as modules, for example `python -m data.download_data`, not `python data/download_data.py`. Running a script by path puts that script's directory on `sys.path` instead of the repository root.

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
