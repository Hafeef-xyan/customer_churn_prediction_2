import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from mlflow.tracking import MlflowClient
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import(
    accuracy_score,
    f1_score,
    precision_score
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / 'data' / 'processed_data' / "initial_train.csv"

MODEL_NAME = 'CustomerChurnModel'
EXPERIMENT_NAME = 'Customer_Churn_Experiments'

"""mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)"""

dagshub.init(
    repo_owner='Hafeef-xyan',
    repo_name="customer_churn_prediction_2",
    mlflow=True
)
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()

# 1. preprocessing
def preprocess():

    # load data
    df = pd.read_csv(DATA_PATH)

    # seperate features and targets
    x = df.drop('Churn',axis=1)
    y = df['Churn']

    # split training and testing data
    x_train , x_test , y_train , y_test = train_test_split(
        x,y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()

    return x_train , x_test, y_train, y_test, scaler

# 2. MODEL TRAINING 
def train_model(x_train, y_train, scaler):

    # model Pipeline
    model = Pipeline([
        ('scaler',scaler),
        ('model',LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # train model
    model.fit(x_train,y_train)

    print("Model Training completed..!!")

    return model

# 3. Evaluating
def evaluate(model,x_test,y_test):

    #predicting
    pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test,pred),
        "f1": f1_score(y_test,pred),
        "precision": precision_score(y_test,pred)
    }

    return metrics

if __name__ == '__main__':

    with mlflow.start_run(run_name="churn_run") as run:

        # 1. preprocessing
        x_train , x_test, y_train, y_test, scaler = preprocess()
    
        # 2. MODEL TRAINING
        model = train_model(x_train, y_train, scaler)

        # log params
        mlflow.log_params({
            "model":"LogisticRegression",
            "max_iter":1000,
            "class_weight":'balanced',
            "random_state":42
        })
    
        # 3. Evaluating
        metrics = evaluate(model,x_test,y_test)

        # log metrics
        mlflow.log_metrics(metrics)
    
        print("\nModel Evaluation Results")
        
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # 4. log model artifact
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="logistic_model",
        )

        # 5. Register Model
        model_uri = model_info.model_uri

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        model_version = registered_model.version

        print("\n model registered successfully.")
        print("model version :",model_version)

        # 6. production promotion
        accuracy = metrics['accuracy']

        if accuracy >= 0.80:

            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias='production',
                version=model_version
            )
            print("model promoted to production.")
        else:
            print("\nModel did not meet the 80% accuracy threshold.")
            print("Model registered, but not promoted to production.")