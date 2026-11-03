import mlflow
import mlflow.sklearn
import pandas as pd
import os
import dagshub

from .schemas import CustomerData
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator

MODEL_NAME = 'CustomerChurnModel'
dagshub.init(
    repo_owner='Hafeef-xyan',
    repo_name='customer_churn_prediction_2',
    mlflow=True
)



client = MlflowClient()

def load_model():

    try:
        model_version = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias='production'
        )

        model = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}@production"
        )

        return model, model_version.version
    except Exception as e:
        raise RuntimeError(
            f"Could not load the Production model: {e}"
        )

model, model_version = load_model()

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using an MLflow registered model."
)

Instrumentator().instrument(app).expose(app)

@app.get("/")
def home():
    return {
        'message':'custmer churn prediction API.',
        'status':"running"
    }

@app.get('/hello')
def hello():
    return {
        "message":"hello"
    }

@app.get('/health')
def health():
    return {
        'status':'healthy',
        'model_loaded':model is not None
    }

@app.get('/model-info')
def model_info():
    return {
        'model_name':MODEL_NAME,
        'model_version':model_version,
        'model_alias':'production'
    }

#prediction endpoint
@app.post('/predict')
def predict(data:CustomerData):

    try:
        input = data.model_dump()

        df = pd.DataFrame([input])

        prediction = int(model.predict(df)[0])

        return {
            'prediction':prediction,
            'model_version':model_version
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )