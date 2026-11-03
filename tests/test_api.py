from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_root():
    response = client.get("/")

    assert response.status_code == 200

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

def test_prediction():

    customer_data = {
        "AccountWeeks": 100,
        "ContractRenewal": 1,
        "DataPlan": 1,
        "DataUsage": 2.5,
        "CustServCalls": 2,
        "DayMins": 180,
        "DayCalls": 100,
        "MonthlyCharge": 75,
        "OverageFee": 10
    }

    response = client.post('/predict',json=customer_data)

    assert response.status_code == 200

    result = response.json()

    assert 'prediction' in result
    assert "model_version" in result

def prediction_invalid_input():

    customer_data = {
        "AccountWeeks": -10,
        "ContractRenewal": 1,
        "DataPlan": 1,
        "DataUsage": 2.5,
        "CustServCalls": 2,
        "DayMins": 180,
        "DayCalls": 100,
        "MonthlyCharge": 75,
        "OverageFee": 10
    }

    response = client.post('/predict',json=customer_data)

    assert response.status_code == 422

def test_hello():
    response = client.get("/hello")

    assert response.status_code == 200