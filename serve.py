from fastapi import FastAPI
import mlflow
from pydantic import BaseModel
import pandas as pd

mlflow.set_tracking_uri("http://mlflow:5000")
model = mlflow.sklearn.load_model("models:/wine-classifier/latest")

app = FastAPI()


class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


@app.post("/predict")
def post_data(user_input: WineFeatures):
    data = pd.DataFrame([user_input.model_dump()])
    predict = model.predict(data)
    return {"prediction": int(predict[0])}
