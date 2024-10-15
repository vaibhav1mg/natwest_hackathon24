import os
import dill  # Use dill for handling custom objects
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from prometheus_client import start_http_server, Summary, Gauge

# Define paths
MODEL_DIR = "../models/"
METRICS_DIR = "../metrics/"

# Initialize Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
MODEL_PREDICTION_GAUGE = Gauge('model_prediction', 'Prediction made by the model', ['model_version', 'prediction'])
METRICS_GAUGE = Gauge('model_metrics', 'Model metrics', ['model_version', 'metric'])

# Function to get the latest model and corresponding files
def get_latest_model_and_files():
    try:
        # Get all models in the directory, sorted by timestamp (assumed in file names)
        model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('model_') and f.endswith('.pkl')],
                             key=lambda x: x.split('_')[1], reverse=True)
        
        if not model_files:
            raise FileNotFoundError("No model files found.")
        
        # Select the latest model file
        latest_model_file = model_files[0]
        model_path = os.path.join(MODEL_DIR, latest_model_file)

        # Load the latest model
        with open(model_path, 'rb') as f:
            model = dill.load(f)
        print(f"Latest model loaded: {latest_model_file}")

        # Load corresponding encoders and scaler
        scaler_file = os.path.join(MODEL_DIR, 'scaler.pkl')
        merchant_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/merchant_encoder.pkl')
        category_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/category_encoder.pkl')
        gender_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/gender_encoder.pkl')
        job_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/job_encoder.pkl')

        with open(scaler_file, 'rb') as f:
            scaler = dill.load(f)
        with open(merchant_encoder_file, 'rb') as f:
            merchant_encoder = dill.load(f)
        with open(category_encoder_file, 'rb') as f:
            category_encoder = dill.load(f)
        with open(gender_encoder_file, 'rb') as f:
            gender_encoder = dill.load(f)
        with open(job_encoder_file, 'rb') as f:
            job_encoder = dill.load(f)

        return model, scaler, merchant_encoder, category_encoder, gender_encoder, job_encoder, latest_model_file
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading latest model: {str(e)}")

# Function to load the latest metrics
def load_latest_metrics():
    try:
        # Get all metrics files, sorted by timestamp
        metrics_files = sorted([f for f in os.listdir(METRICS_DIR) if f.startswith('metrics_') and f.endswith('.csv')],
                               key=lambda x: x.split('_')[1], reverse=True)
        
        if not metrics_files:
            raise FileNotFoundError("No metrics files found.")
        
        # Select the latest metrics file
        latest_metrics_file = metrics_files[0]
        metrics_path = os.path.join(METRICS_DIR, latest_metrics_file)

        # Load the metrics file into a DataFrame
        metrics_df = pd.read_csv(metrics_path)
        latest_metrics = metrics_df.to_dict(orient='records')[0]

        return latest_metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading latest metrics: {str(e)}")

# Load the latest model and related files initially
model, scaler, merchant_encoder, category_encoder, gender_encoder, job_encoder, latest_model_file = get_latest_model_and_files()

# Initialize FastAPI app
app = FastAPI()

# Start Prometheus metrics server on port 8001
start_http_server(8001)

# Define the input schema for the API using Pydantic
class TransactionData(BaseModel):
    merchant: str
    category: str
    amt: float
    gender: str
    job: str
    lat: float
    long: float
    city_pop: int
    merch_lat: float
    merch_long: float
    unix_time: int

# Preprocessing function
def preprocess(data: TransactionData):
    try:
        # Encode categorical features
        merchant_encoded = merchant_encoder.transform([data.merchant])[0]
        category_encoded = category_encoder.transform([data.category])[0]
        gender_encoded = gender_encoder.transform([data.gender])[0]
        job_encoded = job_encoder.transform([data.job])[0]

        # Scale numeric features
        amt_scaled = scaler.transform(np.array([[data.amt]]))[0][0]

        # Combine all features into a single input vector, including unix_time
        features = np.array([
            merchant_encoded, category_encoded, amt_scaled,
            gender_encoded, job_encoded, data.lat, data.long,
            data.city_pop, data.merch_lat, data.merch_long, data.unix_time
        ]).reshape(1, -1)

        return features
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

# Define the API route for prediction
@app.post("/predict/")
async def predict(data: TransactionData):
    try:
        # Preprocess the input data
        features = preprocess(data)

        # Predict the outcome
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        # Update metrics gauge
        MODEL_PREDICTION_GAUGE.labels(model_version=latest_model_file, prediction=int(prediction[0])).set(probability)

        return {
            "prediction": int(prediction[0]),
            "probability_of_fraud": probability,
            "model_version": latest_model_file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Define the API route for metrics
@app.get("/metrics/")
async def get_metrics():
    try:
        # Load the latest metrics
        latest_metrics = load_latest_metrics()

        # Update the metrics gauge with values from the latest metrics
        for metric, value in latest_metrics.items():
            if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                METRICS_GAUGE.labels(model_version=latest_model_file, metric=metric).set(value)

        return latest_metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

# Define the API route for model information
@app.get("/model_info/")
async def get_model_info():
    try:
        # Return the current model version and any other relevant information
        return {
            "model_version": latest_model_file,
            "status": "active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
