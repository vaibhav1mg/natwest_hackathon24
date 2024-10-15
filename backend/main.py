# backend/main.py

from contextlib import asynccontextmanager
import csv
import os
import dill  # Use dill for handling custom objects
from matplotlib import pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from prometheus_client import start_http_server, Summary, Gauge, Counter, Histogram
import time
import logging
import requests
from scipy.stats import ks_2samp
import json
import shap
import datetime
from typing import List, Optional
import psutil
from fastapi import Query

# Ensure the logs directory exists
log_dir = '../logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging to write JSON logs
logging.basicConfig(
    filename=os.path.join(log_dir, 'transactions.log'),
    level=logging.INFO,
    format='%(message)s'
)

# Define paths
MODEL_DIR = "../models/"
METRICS_DIR = "../metrics/"
DATASET_DIR = "../datasets/"
ASSETS_DIR = "../assets/"


# Initialize Prometheus metrics
REQUEST_TIME = Histogram('request_processing_seconds', 'Time spent processing request')
MODEL_PREDICTION_GAUGE = Gauge('model_prediction', 'Prediction made by the model', ['model_version', 'prediction'])
METRICS_GAUGE = Gauge('model_metrics', 'Model metrics', ['model_version', 'metric'])
TOTAL_REQUESTS = Counter('total_prediction_requests', 'Total number of prediction requests')
SUCCESSFUL_PREDICTIONS = Counter('successful_predictions', 'Number of successful predictions')
FAILED_PREDICTIONS = Counter('failed_predictions', 'Number of failed predictions')
RESPONSE_TIME = Histogram('response_time_seconds', 'Response time for prediction requests')
THROUGHPUT_COUNTER = Counter('throughput_requests', 'Number of requests processed')
ALERTMANAGER_URL = "http://localhost:9093/api/v1/alerts"  # Update if Alertmanager is hosted elsewhere

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


def send_alert(alert_name: str, description: str, severity: str = 'critical'):
    """
    Sends an alert to Alertmanager.

    Args:
        alert_name (str): The name of the alert.
        description (str): A detailed description of the alert.
        severity (str): The severity level of the alert (e.g., 'critical', 'warning').
    """
    alert = [
        {
            "labels": {
                "alertname": alert_name,
                "severity": severity
            },
            "annotations": {
                "description": description
            }
        }
    ]
    try:
        response = requests.post(ALERTMANAGER_URL, json=alert)
        if response.status_code == 200:
            logging.info(f"Alert '{alert_name}' sent successfully.")
        else:
            logging.error(f"Failed to send alert '{alert_name}': {response.text}")
    except Exception as e:
        logging.error(f"Error sending alert '{alert_name}': {str(e)}")



# Function to get the latest model and corresponding files
def get_latest_model_and_files():
    try:
        # Get all models in the directory, sorted by timestamp (assumed in file names)
        model_files = sorted(
            [f for f in os.listdir(MODEL_DIR) if f.startswith('model_') and f.endswith('.pkl')],
            key=lambda x: x.split('_')[1],
            reverse=True
        )

        if not model_files:
            raise FileNotFoundError("No model files found.")

        # Select the latest model file
        latest_model_file = model_files[0]
        model_path = os.path.join(MODEL_DIR, latest_model_file)

        # Load the latest model
        with open(model_path, 'rb') as f:
            model = dill.load(f)
        logging.info(f"Latest model loaded: {latest_model_file}")

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
        logging.error(f"Error loading latest model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading latest model: {str(e)}")

# Function to load the latest metrics
def load_latest_metrics():
    try:
        # Get all metrics files, sorted by timestamp
        metrics_files = sorted(
            [f for f in os.listdir(METRICS_DIR) if f.startswith('metrics_') and f.endswith('.csv')],
            key=lambda x: x.split('_')[1],
            reverse=True
        )

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
        logging.error(f"Error loading latest metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading latest metrics: {str(e)}")

# Function to load a specific model
def load_model(model_version):
    try:
        global model, scaler, merchant_encoder, category_encoder, gender_encoder, job_encoder, latest_model_file

        model_path = os.path.join(MODEL_DIR, model_version)
        with open(model_path, 'rb') as f:
            loaded_model = dill.load(f)
        logging.info(f"Model {model_version} loaded successfully.")

        # Load encoders and scaler
        scaler_file = os.path.join(MODEL_DIR, 'scaler.pkl')
        merchant_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/merchant_encoder.pkl')
        category_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/category_encoder.pkl')
        gender_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/gender_encoder.pkl')
        job_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/job_encoder.pkl')

        with open(scaler_file, 'rb') as f:
            loaded_scaler = dill.load(f)
        with open(merchant_encoder_file, 'rb') as f:
            loaded_merchant_encoder = dill.load(f)
        with open(category_encoder_file, 'rb') as f:
            loaded_category_encoder = dill.load(f)
        with open(gender_encoder_file, 'rb') as f:
            loaded_gender_encoder = dill.load(f)
        with open(job_encoder_file, 'rb') as f:
            loaded_job_encoder = dill.load(f)

        model, scaler, merchant_encoder, category_encoder, gender_encoder, job_encoder, latest_model_file = (
            loaded_model, loaded_scaler, loaded_merchant_encoder,
            loaded_category_encoder, loaded_gender_encoder, loaded_job_encoder, model_version
        )

        # Re-initialize SHAP explainer with the new model
        global explainer
        explainer = shap.TreeExplainer(model)

    except Exception as e:
        logging.error(f"Error loading model {model_version}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model {model_version}: {str(e)}")

# Load the latest model and related files initially
model, scaler, merchant_encoder, category_encoder, gender_encoder, job_encoder, latest_model_file = get_latest_model_and_files()

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Initialize FastAPI app
app = FastAPI()

# Start Prometheus metrics server on port 8001
start_http_server(8001)

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
        logging.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

# Function to detect anomalies (simple rule-based)
def detect_anomaly(data: TransactionData):
    # Example rules for anomaly detection
    if data.amt > 1000:
        return True
    if data.merchant.lower().startswith("fraud"):
        return True
    return False

# Function to log transactions
def log_transaction(data: TransactionData, prediction: Optional[int], probability: Optional[float], response_time: float, is_anomaly: bool = False):
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "transaction_id": data.unix_time,
        "merchant": data.merchant,
        "category": data.category,
        "amt": data.amt,
        "gender": data.gender,
        "job": data.job,
        "lat": data.lat,
        "long": data.long,
        "city_pop": data.city_pop,
        "merch_lat": data.merch_lat,
        "merch_long": data.merch_long,
        "unix_time": data.unix_time,
        "prediction": prediction,
        "probability_of_fraud": probability,
        "response_time": response_time,
        "anomaly": is_anomaly
    }
    logging.info(json.dumps(log_entry))

# Define the API route for prediction
@app.post("/predict/")
@REQUEST_TIME.time()
def predict(data: TransactionData):
    TOTAL_REQUESTS.inc()  # Increment total requests counter
    THROUGHPUT_COUNTER.inc()  # Increment throughput counter
    start_time = time.time()
    try:
        # Preprocess the input data
        features = preprocess(data)

        # Predict the outcome
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        # Update metrics gauge
        MODEL_PREDICTION_GAUGE.labels(model_version=latest_model_file, prediction=int(prediction[0])).set(float(probability))

        # Update success or failure counters
        if prediction is not None:
            SUCCESSFUL_PREDICTIONS.inc()
        else:
            FAILED_PREDICTIONS.inc()

        # Detect anomalies
        is_anomaly = detect_anomaly(data)

        # Log transaction
        response_time = time.time() - start_time
        log_transaction(data, int(prediction[0]), float(probability), response_time, is_anomaly)

        # Update response time histogram
        RESPONSE_TIME.observe(response_time)

        return {
            "prediction": int(prediction[0]),
            "probability_of_fraud": float(probability),
            "model_version": latest_model_file,
            "anomaly": is_anomaly
        }
    except Exception as e:
        FAILED_PREDICTIONS.inc()
        response_time = time.time() - start_time
        log_transaction(data, None, None, response_time, False)
        RESPONSE_TIME.observe(response_time)
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Define the API route for metrics
@app.get("/metrics/")
def get_metrics():
    try:
        # Load the latest metrics
        latest_metrics = load_latest_metrics()

        # Update the metrics gauge with values from the latest metrics
        for metric, value in latest_metrics.items():
            if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                try:
                    METRICS_GAUGE.labels(model_version=latest_model_file, metric=metric).set(float(value))
                except ValueError:
                    logging.warning(f"Could not convert value to float for metric '{metric}': {value}")

        # Convert all metric values to native Python types, handling non-numeric values
        serializable_metrics = {}
        for k, v in latest_metrics.items():
            try:
                serializable_metrics[k] = float(v)
            except ValueError:
                logging.warning(f"Skipping non-numeric metric '{k}' with value: {v}")
                serializable_metrics[k] = v  # Optionally store the original value

        return serializable_metrics
    except Exception as e:
        logging.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

# Define the API route for model information
@app.get("/model_info/")
def get_model_info():
    try:
        # Return the current model version and any other relevant information
        return {
            "model_version": latest_model_file,
            "status": "active"
        }
    except Exception as e:
        logging.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


from prometheus_client import REGISTRY

@app.get("/simulation_metrics/")
def get_simulation_metrics():
    try:
        # Assuming you're using Prometheus-style metrics
        total_requests_count = TOTAL_REQUESTS._value.get()
        successful_predictions_count = SUCCESSFUL_PREDICTIONS._value.get()
        failed_predictions_count = FAILED_PREDICTIONS._value.get()

        # Extract the histogram samples
        histogram_data = RESPONSE_TIME.collect()[0].samples
        response_time_count = None
        response_time_sum = None

        # Iterate over the samples to get count and sum
        for sample in histogram_data:
            if sample.name.endswith("_count"):
                response_time_count = sample.value
            elif sample.name.endswith("_sum"):
                response_time_sum = sample.value

        # Make sure we got both values
        if response_time_count is None or response_time_sum is None:
            raise ValueError("Histogram data is missing count or sum.")

        # Calculate average response time
        average_response_time = response_time_sum / response_time_count if response_time_count > 0 else 0.0

        # Throughput calculation
        time_window_seconds = 60  # Example: last 60 seconds
        throughput = total_requests_count / time_window_seconds

        return {
            "total_requests": int(total_requests_count),
            "successful_predictions": int(successful_predictions_count),
            "failed_predictions": int(failed_predictions_count),
            "average_response_time": float(average_response_time),
            "throughput": float(throughput)
        }
    except Exception as e:
        logging.error(f"Error retrieving simulation metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving simulation metrics: {str(e)}")


# Define the API route for data drift detection
@app.get("/data_drift/")
def data_drift():
    try:
        # Load training and test data
        train_df = pd.read_csv(os.path.join(DATASET_DIR, "fraudTrain.csv")).sample(n=30000, random_state=42).reset_index(drop=True)
        test_df = pd.read_csv(os.path.join(DATASET_DIR, "fraudTest.csv")).sample(n=5000, random_state=42).reset_index(drop=True)

        # Verify if 'time' or 'unix_time' exists
        if 'time' in train_df.columns and 'time' in test_df.columns:
            numerical_features = ['amt', 'time']
        elif 'unix_time' in train_df.columns and 'unix_time' in test_df.columns:
            numerical_features = ['amt', 'unix_time']
        else:
            raise KeyError("'time' or 'unix_time' column is missing in the datasets.")

        drift_scores = {}
        for feature in numerical_features:
            statistic, p_value = ks_2samp(train_df[feature], test_df[feature])
            drift_scores[feature] = {
                "ks_statistic": statistic,
                "p_value": p_value
            }

        logging.info(f"Data drift scores: {drift_scores}")

        # Trigger alert if any feature shows significant drift
        for feature, scores in drift_scores.items():
            if scores["p_value"] < 0.05:
                alert_name = "DataDriftDetected"
                description = f"Significant data drift detected in feature '{feature}' with KS statistic {scores['ks_statistic']:.4f} and p-value {scores['p_value']:.4f}."
                send_alert(alert_name, description, severity='critical')

        return drift_scores
    except KeyError as ke:
        logging.error(f"Data Drift Detection Key Error: {str(ke)}")
        raise HTTPException(status_code=500, detail=f"Data Drift Detection Key Error: {str(ke)}")
    except Exception as e:
        logging.error(f"Data Drift Detection Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data Drift Detection Error: {str(e)}")


# Define the API route to list all available models
@app.get("/list_models/")
def list_models():
    try:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('model_') and f.endswith('.pkl')]
        model_files_sorted = sorted(model_files, reverse=True)  # Latest first
        return {"available_models": model_files_sorted}
    except Exception as e:
        logging.error(f"List Models Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"List Models Error: {str(e)}")

# Define the API route to activate a specific model version
@app.post("/activate_model/{model_version}/")
def activate_model(model_version: str):
    try:
        global model, scaler, merchant_encoder, category_encoder, gender_encoder, job_encoder, latest_model_file

        # Validate model existence
        model_path = os.path.join(MODEL_DIR, model_version)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_version} not found.")

        # Load and set the new active model
        load_model(model_version)

        logging.info(f"Model {model_version} activated successfully.")
        return {"message": f"Model {model_version} activated successfully."}
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Activate Model Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Activate Model Error: {str(e)}")

# Define the API route to rollback to the previous model version
@app.post("/rollback/")
def rollback_model():
    try:
        global model, scaler, merchant_encoder, category_encoder, gender_encoder, job_encoder, latest_model_file

        model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('model_') and f.endswith('.pkl')]
        model_files_sorted = sorted(model_files, reverse=True)  # Latest first

        if len(model_files_sorted) < 2:
            raise HTTPException(status_code=400, detail="No previous model version available to rollback.")

        current_model = latest_model_file
        previous_model = model_files_sorted[1]  # Second latest

        # Activate the previous model
        load_model(previous_model)

        logging.info(f"Rolled back to model {previous_model} successfully.")
        return {"message": f"Rolled back to model {previous_model} successfully."}
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Rollback Model Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rollback Model Error: {str(e)}")


# Preprocessing function
def preprocess_dataset_for_shap(df):
    try:
        logging.info("Preprocessing dataset for SHAP...")
        
        # Drop any unnecessary or non-numeric columns
        columns_to_drop = [
            'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last',
            'street', 'city', 'state', 'zip', 'dob', 'trans_num'
        ]
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Handle categorical columns
        categorical_columns = ['merchant', 'category', 'gender', 'job']
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes  # Convert categories to numeric codes
        
        logging.info("Dataset successfully preprocessed.")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing for SHAP: {str(e)}")
        return None

# Preprocessing function
def preprocess_dataset_for_shap(df):
    try:
        logging.info("Preprocessing dataset for SHAP...")
        
        # Drop any unnecessary or non-numeric columns
        columns_to_drop = [
            'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last',
            'street', 'city', 'state', 'zip', 'dob', 'trans_num'
        ]
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Handle categorical columns
        categorical_columns = ['merchant', 'category', 'gender', 'job']
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes  # Convert categories to numeric codes
        
        logging.info("Dataset successfully preprocessed.")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing for SHAP: {str(e)}")
        return None


@app.get("/transaction/{transaction_id}/")
def get_transaction(transaction_id: int):
    try:
        log_path = os.path.join(log_dir, 'transactions.log')
        if not os.path.exists(log_path):
            raise FileNotFoundError("Transaction log file not found.")

        with open(log_path, "r") as log_file:
            logs = log_file.readlines()

        # Search for the transaction by ID
        for log in reversed(logs):  # Start from the latest
            try:
                transaction = json.loads(log.strip())
                if transaction.get("transaction_id") == transaction_id:
                    return transaction
            except json.JSONDecodeError:
                continue

        raise HTTPException(status_code=404, detail=f"Transaction ID {transaction_id} not found.")
    except FileNotFoundError as fe:
        logging.error(str(fe))
        raise HTTPException(status_code=500, detail=str(fe))
    except Exception as e:
        logging.error(f"Get Transaction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Get Transaction Error: {str(e)}")

    




@app.get("/filtered_logs/")
def get_filtered_logs(
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    transaction_id: Optional[int] = Query(None, description="Transaction ID to filter"),
    anomaly: Optional[bool] = Query(None, description="Filter by anomaly status"),
    prediction: Optional[int] = Query(None, description="Filter by prediction outcome (0 or 1)"),
    merchant: Optional[str] = Query(None, description="Filter by merchant name")
):
    try:
        log_path = os.path.join(log_dir, 'transactions.log')
        if not os.path.exists(log_path):
            raise FileNotFoundError("Transaction log file not found.")

        with open(log_path, "r") as log_file:
            logs = log_file.readlines()

        filtered_transactions = []
        for log in logs:
            try:
                transaction = json.loads(log.strip())
                
                # Filter by start and end date
                if start_date:
                    transaction_time = datetime.fromisoformat(transaction.get("timestamp"))
                    if transaction_time < datetime.fromisoformat(start_date):
                        continue
                if end_date:
                    transaction_time = datetime.fromisoformat(transaction.get("timestamp"))
                    if transaction_time > datetime.fromisoformat(end_date):
                        continue

                # Filter by transaction_id
                if transaction_id is not None and transaction.get("transaction_id") != transaction_id:
                    continue

                # Filter by anomaly
                if anomaly is not None and transaction.get("anomaly") != anomaly:
                    continue

                # Filter by prediction
                if prediction is not None and transaction.get("prediction") != prediction:
                    continue

                # Filter by merchant
                if merchant is not None and merchant.lower() not in transaction.get("merchant", "").lower():
                    continue

                filtered_transactions.append(transaction)
            except json.JSONDecodeError:
                continue

        return {"filtered_transactions": filtered_transactions}
    except FileNotFoundError as fe:
        logging.error(str(fe))
        raise HTTPException(status_code=500, detail=str(fe))
    except Exception as e:
        logging.error(f"Filtered Logs Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Filtered Logs Error: {str(e)}")




@app.get("/metrics/")
def get_metrics():
    try:
        # Load the latest metrics
        latest_metrics = load_latest_metrics()

        # Update the metrics gauge with values from the latest metrics
        for metric, value in latest_metrics.items():
            if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                try:
                    METRICS_GAUGE.labels(model_version=latest_model_file, metric=metric).set(float(value))
                except ValueError:
                    logging.warning(f"Could not convert value to float for metric '{metric}': {value}")

        # Define thresholds
        METRIC_THRESHOLDS = {
            "accuracy": 0.95,
            "precision": 0.70,
            "recall": 0.75
        }

        # Check metrics against thresholds and send alerts if necessary
        for metric, threshold in METRIC_THRESHOLDS.items():
            metric_value = float(latest_metrics.get(metric, 1.0))
            if metric_value < threshold:
                alert_name = f"{metric.capitalize()}BelowThreshold"
                description = f"Model {latest_model_file} has {metric} of {metric_value:.2f}, which is below the threshold of {threshold:.2f}."
                send_alert(alert_name, description, severity='warning')

        # Convert all metric values to native Python types, handling non-numeric values
        serializable_metrics = {}
        for k, v in latest_metrics.items():
            try:
                serializable_metrics[k] = float(v)
            except ValueError:
                logging.warning(f"Skipping non-numeric metric '{k}' with value: {v}")
                serializable_metrics[k] = v  # Optionally store the original value

        return serializable_metrics
    except Exception as e:
        logging.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")


@app.get("/system_metrics/")
def get_system_metrics():
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        return {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory_info.percent,
            "disk_usage_percent": disk_info.percent
        }
    except Exception as e:
        logging.error(f"System Metrics Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System Metrics Error: {str(e)}")
    

@app.post("/log_alert/")
async def log_alert(alerts: List[dict]):
    try:
        with open(os.path.join(log_dir, 'alerts.log'), 'a') as f:
            for alert in alerts:
                f.write(json.dumps(alert) + "\n")
        return {"status": "success", "message": "Alerts logged successfully."}
    except Exception as e:
        logging.error(f"Logging Alert Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Logging Alert Error: {str(e)}")




class ModelDetail(BaseModel):
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    timestamp: str
    train_data_path: str
    test_data_path: str
    hyperparameters: str

@app.get("/model_details/", response_model=List[ModelDetail])
def get_model_details():
    try:
        metrics_files = [f for f in os.listdir(METRICS_DIR) if f.startswith('metrics_') and f.endswith('.csv')]
        model_details = []
        
        for file in metrics_files:
            metrics_path = os.path.join(METRICS_DIR, file)
            with open(metrics_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    model_detail = ModelDetail(
                        model_version=row.get('model_version', 'N/A'),
                        accuracy=float(row.get('accuracy', 0.0)),
                        precision=float(row.get('precision', 0.0)),
                        recall=float(row.get('recall', 0.0)),
                        f1_score=float(row.get('f1_score', 0.0)),
                        roc_auc=float(row.get('roc_auc', 0.0)),
                        timestamp=row.get('timestamp', 'N/A'),
                        train_data_path=row.get('train_data_path', 'N/A'),
                        test_data_path=row.get('test_data_path', 'N/A'),
                        hyperparameters=row.get('hyperparameters', 'N/A')
                    )
                    model_details.append(model_detail)
        return model_details
    except Exception as e:
        logging.error(f"Error fetching model details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching model details: {str(e)}")

    
    
# SHAP Summary and Dependence Plot Generation Function
def generate_shap_plots(features, shap_values, feature_names, dependence_feature="amt", model_name="latest_model"):
    try:
        logging.info("Generating SHAP summary and dependence plots...")
        
        # Determine the correct SHAP values to use
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values_to_use = shap_values[1]  # Multi-class: use class 1
                logging.info("Using SHAP values for class 1.")
            else:
                shap_values_to_use = shap_values[0]  # Binary or single-output
                logging.info("Using SHAP values for class 0.")
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3 and shap_values.shape[2] == 2:
                shap_values_to_use = shap_values[:, :, 1]  # Binary classifier with 3D array
                logging.info("Using SHAP values for class 1 (binary classifier).")
            elif shap_values.ndim == 2:
                shap_values_to_use = shap_values  # Single-output model
                logging.info("Using SHAP values as is.")
            else:
                shap_values_to_use = shap_values  # Default
                logging.warning(f"Unexpected SHAP values shape: {shap_values.shape}")
        else:
            shap_values_to_use = shap_values  # Default
            logging.warning("SHAP values are neither a list nor a numpy array. Using as is.")
        
        # Log shapes for debugging
        logging.info(f"Features shape: {features.shape}")
        logging.info(f"SHAP values to use shape: {shap_values_to_use.shape}")
        
        # Create model-specific assets directory
        model_assets_dir = os.path.join(ASSETS_DIR, model_name)
        if not os.path.exists(model_assets_dir):
            os.makedirs(model_assets_dir)
            logging.info(f"Created assets directory at {model_assets_dir}.")
        
        # Generate the SHAP summary plot
        shap_summary_path = os.path.join(model_assets_dir, "shap_summary.png")
        shap.summary_plot(shap_values_to_use, features, feature_names=feature_names, show=False)
        plt.savefig(shap_summary_path)
        plt.close()
        logging.info(f"SHAP summary plot saved at: {shap_summary_path}")
        
        # Generate the SHAP dependence plot for the selected feature
        shap_dependence_path = os.path.join(model_assets_dir, f"shap_dependence_{dependence_feature}.png")
        shap.dependence_plot(dependence_feature, shap_values_to_use, features, feature_names=feature_names, show=False)
        plt.savefig(shap_dependence_path)
        plt.close()
        logging.info(f"SHAP dependence plot saved at: {shap_dependence_path}")
        
        return shap_summary_path, shap_dependence_path
    except Exception as e:
        logging.error(f"Error generating SHAP plots: {str(e)}")
        return None, None






# Endpoint to generate SHAP summary and dependence plots manually
@app.get("/generate_shap_plots/")
async def generate_shap_plots_endpoint(model_name: str, dependence_feature: str = "amt"):
    try:
        logging.info(f"Triggering SHAP plots generation for model {model_name} and feature '{dependence_feature}'...")
        
        # Check if the model exists
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            logging.error(f"Model file {model_name} does not exist.")
            return {"status": "error", "message": f"Model {model_name} does not exist."}
        
        # Load the specified model
        with open(model_path, 'rb') as f:
            loaded_model = dill.load(f)
        logging.info(f"Loaded model: {model_name}")
        
        # Load encoders and scaler
        scaler_file = os.path.join(MODEL_DIR, 'scaler.pkl')
        merchant_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/merchant_encoder.pkl')
        category_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/category_encoder.pkl')
        gender_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/gender_encoder.pkl')
        job_encoder_file = os.path.join(MODEL_DIR, 'label_encoders/job_encoder.pkl')
        
        with open(scaler_file, 'rb') as f:
            loaded_scaler = dill.load(f)
        with open(merchant_encoder_file, 'rb') as f:
            loaded_merchant_encoder = dill.load(f)
        with open(category_encoder_file, 'rb') as f:
            loaded_category_encoder = dill.load(f)
        with open(gender_encoder_file, 'rb') as f:
            loaded_gender_encoder = dill.load(f)
        with open(job_encoder_file, 'rb') as f:
            loaded_job_encoder = dill.load(f)
        
        # Initialize SHAP explainer for the specified model
        loaded_explainer = shap.TreeExplainer(loaded_model)
        logging.info(f"SHAP explainer initialized for model {model_name}.")
        
        # Load and preprocess the dataset
        test_data_path = os.path.join(DATASET_DIR, "fraudTest.csv")
        if not os.path.exists(test_data_path):
            logging.error(f"Dataset file not found at {test_data_path}.")
            return {"status": "error", "message": "Dataset file not found."}
        
        test_data = pd.read_csv(test_data_path).sample(1000, random_state=42)
        X_test = preprocess_dataset_for_shap(test_data.drop(columns=["is_fraud"]))
        
        if X_test is None:
            logging.error("Dataset preprocessing failed.")
            return {"status": "error", "message": "Dataset preprocessing failed."}
        
        logging.info(f"Dataset shape after preprocessing: {X_test.shape}")
        
        # Ensure the dataset has valid feature names
        X_test.columns = X_test.columns.astype(str)
        
        # Compute SHAP values for the entire test dataset
        logging.info("Computing SHAP values...")
        shap_values = loaded_explainer.shap_values(X_test)
        
        # Generate and save SHAP summary and dependence plots
        shap_summary_path, shap_dependence_path = generate_shap_plots(
            X_test,
            shap_values,
            X_test.columns.tolist(),
            dependence_feature=dependence_feature,
            model_name=model_name
        )
        
        if shap_summary_path and shap_dependence_path:
            return {
                "status": "success",
                "message": f"SHAP plots generated for model {model_name}.",
                "summary_plot": shap_summary_path,
                "dependence_plot": shap_dependence_path
            }
        else:
            return {"status": "error", "message": "Failed to generate SHAP plots."}
        
    except Exception as e:
        logging.error(f"Error during SHAP plots generation: {str(e)}")
        return {"status": "error", "message": f"Error during SHAP plots generation: {str(e)}"}

# Main function to start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    # Load the latest model upon startup
    model, scaler, merchant_encoder, category_encoder, gender_encoder, job_encoder, latest_model_file = get_latest_model_and_files()
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    logging.info("SHAP explainer initialized successfully.")
    uvicorn.run(app, host="127.0.0.1", port=8000)