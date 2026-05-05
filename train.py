from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pandas as pd
import mlflow
import os

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

data = load_wine()
MAX_DEPTH = 2

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.33, random_state=42
)


clf = RandomForestClassifier(max_depth=MAX_DEPTH, random_state=0)

with mlflow.start_run():
    clf.fit(X_train, y_train)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_metric("accuracy", clf.score(X_test, y_test))
    mlflow.sklearn.log_model(clf, "model")
