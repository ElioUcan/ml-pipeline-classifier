# ML Pipeline — Wine Classifier

## 🛠️ Technologies
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

## ✨ Features
- Train a `RandomForestClassifier` on the scikit-learn Wine dataset with experiment tracking via MLflow
- Serve the latest registered model through a `POST /predict` FastAPI endpoint
- MLflow tracking server persists experiments, runs, and model artifacts via volume mounts
- GitHub Actions retrains the model and uploads artifacts on every push to `main`
- Docker Compose orchestrates both services with a health-check dependency — the API waits for MLflow before starting

## 🎯 Uses
End-to-end MLOps reference project demonstrating how to train, version, serve, and automatically retrain a machine learning model. Built as project #4 in a Data/AI/MLOps engineering portfolio.

## 🔧 Process
`train.py` logs hyperparameters, accuracy, and the serialized model to MLflow after each run. `serve.py` loads `models:/wine-classifier/latest` from the registry and exposes `/predict`. Docker Compose runs both services; the API container uses a health check on the MLflow container before starting. The GitHub Actions workflow runs `train.py` on every push to `main` and uploads artifacts for traceability.

## 💡 Learnings
- MLflow's model registry (`models:/name/latest`) decouples training from serving — the API doesn't need to know where or when the model was trained
- Docker health checks are the right way to handle service startup ordering in Compose; `depends_on` alone is not enough
- GitHub Actions can act as a minimal retraining scheduler with artifact storage at no extra infrastructure cost

## ▶️ Running the project

```bash
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |
| Prediction API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
