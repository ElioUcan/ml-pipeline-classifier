# ML Pipeline — Wine Classifier

An end-to-end machine learning pipeline that trains a wine classification model, tracks experiments with MLflow, and serves predictions through a FastAPI REST API. The entire stack runs via Docker Compose, and retraining is automated on every push to `main` through GitHub Actions.

---

## Project Structure

```
project-4-ml-pipeline/
├── train.py                     # Model training script
├── serve.py                     # FastAPI inference server
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Orchestrates MLflow + API services
├── Dockerfile.api               # Container for the FastAPI server
├── Dockerfile.mlflow            # Container for the MLflow tracking server
├── .env.example                 # Environment variable template
├── .gitignore
└── .github/
    └── workflows/
        └── retrain.yml          # CI/CD retraining workflow
```

---

## How It Works

### 1. Training (`train.py`)

- Loads the [Wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset) from scikit-learn.
- Trains a `RandomForestClassifier` with `max_depth=2` using an 67/33 train/test split.
- Logs the hyperparameter (`max_depth`), accuracy metric, and the serialized model to MLflow.
- The MLflow tracking URI defaults to `sqlite:///mlflow.db` but can be overridden via the `MLFLOW_TRACKING_URI` environment variable.

### 2. Serving (`serve.py`)

- Loads the latest registered version of the `wine-classifier` model from MLflow (`models:/wine-classifier/latest`).
- Exposes a `POST /predict` endpoint via FastAPI.
- Accepts the 13 wine features as a JSON body and returns the predicted class (0, 1, or 2).

### 3. Infrastructure (`docker-compose.yml`)

Two services are defined:

| Service  | Image           | Port | Description                        |
|----------|-----------------|------|------------------------------------|
| `mlflow` | Dockerfile.mlflow | 5000 | MLflow tracking & model registry  |
| `api`    | Dockerfile.api    | 8000 | FastAPI prediction server          |

The `api` service waits for the `mlflow` service to pass its health check before starting.

MLflow persists data via three volume mounts: `mlflow.db`, `mlruns/`, and `mlartifacts/`.

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for local runs without Docker)

### Environment Variables

Copy `.env.example` and fill in your values:

```bash
cp .env.example .env
```

| Variable              | Default               | Description                          |
|-----------------------|-----------------------|--------------------------------------|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | URI for the MLflow tracking server   |

### Local Setup (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &

# Train the model
MLFLOW_TRACKING_URI=http://localhost:5000 python train.py

# Register the model in MLflow UI (http://localhost:5000), then start the API. This is only neccesary the first time.
uvicorn serve:app --host 0.0.0.0 --port 8000
```

### Docker Compose Setup

```bash
docker compose up --build
```

- MLflow UI: http://localhost:5000
- API: http://localhost:8000

---

## API Reference

### `POST /predict`

Predicts the wine class from its chemical features.

**Request body:**

```json
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100.0,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.4,
  "proline": 1050.0
}
```

**Response:**

```json
{
  "prediction": 0
}
```

The prediction is an integer — `0`, `1`, or `2` — corresponding to one of the three wine classes in the dataset.

---

## CI/CD — Automated Retraining

The GitHub Actions workflow (`.github/workflows/retrain.yml`) triggers on every push to `main`:

1. Checks out the code and sets up Python 3.12.
2. Installs dependencies.
3. Starts a local MLflow server.
4. Runs `train.py` to retrain the model.
5. Uploads the model artifacts (`mlartifacts/`) as a GitHub Actions artifact named `wine-classifier-model`.

---

## Dependencies

| Package         | Purpose                              |
|-----------------|--------------------------------------|
| `scikit-learn`  | Model training (RandomForestClassifier) |
| `mlflow`        | Experiment tracking & model registry |
| `fastapi`       | REST API framework                   |
| `uvicorn`       | ASGI server for FastAPI              |
| `pandas`        | Data manipulation                    |
| `python-dotenv` | Environment variable management      |
