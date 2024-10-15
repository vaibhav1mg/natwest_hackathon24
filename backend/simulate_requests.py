import os
import pandas as pd
import requests
import asyncio
import aiohttp
import time
import random
from datetime import datetime
import argparse

# Configuration
PREDICT_ENDPOINT = "http://127.0.0.1:8000/predict/"
REQUEST_INTERVAL = 1  # Default interval in seconds
VARIABILITY_FACTOR = 0.5  # Adjust the variability in request intervals (0 to 1)

# Load and preprocess data
def load_data(path, truncate_size=100000, oversample_fraud=False):
    try:
        data = pd.read_csv(path)
        print(f"Original dataset shape: {data.shape}")

        # Truncate to the desired size
        if data.shape[0] > truncate_size:
            data = data.sample(n=truncate_size, random_state=42).reset_index(drop=True)
            print(f"Dataset truncated to: {data.shape}")
        else:
            print("Dataset size is within the limit. No truncation applied.")

        if oversample_fraud:
            # Separate fraud and non-fraud transactions
            fraud_data = data[data['is_fraud'] == 1]
            non_fraud_data = data[data['is_fraud'] == 0]

            print(f"Fraud transactions: {fraud_data.shape[0]}")
            print(f"Non-fraud transactions: {non_fraud_data.shape[0]}")

            # Oversample fraud transactions to increase their representation
            if fraud_data.empty:
                print("No fraudulent transactions found to oversample.")
            else:
                oversample_ratio = 10  # Adjust as needed
                fraud_oversampled = fraud_data.sample(n=min(len(fraud_data) * oversample_ratio, truncate_size), replace=True, random_state=42)
                data = pd.concat([non_fraud_data, fraud_oversampled], ignore_index=True)
                print(f"After oversampling, dataset shape: {data.shape}")

        # Shuffle the dataset
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Prepare transaction data for prediction
def prepare_transaction(row):
    transaction = {
        "merchant": row["merchant"],
        "category": row["category"],
        "amt": float(row["amt"]),
        "gender": row["gender"],
        "job": row["job"],
        "lat": float(row["lat"]),
        "long": float(row["long"]),
        "city_pop": int(row["city_pop"]),
        "merch_lat": float(row["merch_lat"]),
        "merch_long": float(row["merch_long"]),
        "unix_time": int(row["unix_time"])
    }
    return transaction

# Send prediction request asynchronously
async def send_prediction_async(session, transaction):
    try:
        async with session.post(PREDICT_ENDPOINT, json=transaction) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Prediction request failed with status code {response.status}: {await response.text()}")
                return None
    except Exception as e:
        print(f"Error sending prediction request: {e}")
        return None

# Asynchronous simulation loop
async def simulate_requests_async(data, base_interval=1, variability=0.5):
    total_requests = len(data)
    print(f"Starting asynchronous simulation of {total_requests} requests with a base interval of {base_interval} seconds.")
    
    async with aiohttp.ClientSession() as session:
        for idx, row in data.iterrows():
            transaction = prepare_transaction(row)
            response = await send_prediction_async(session, transaction)
            
            if response:
                print(f"[{datetime.now()}] Request {idx+1}/{total_requests} - Prediction: {response['prediction']}, Probability of Fraud: {response['probability_of_fraud']:.2f}, Model Version: {response['model_version']}")
            
            # Vary the request interval and potentially send bursts of requests
            adjusted_interval = base_interval + random.uniform(-variability, variability)
            adjusted_interval = max(0.1, adjusted_interval)  # Ensure the interval is at least 0.1 seconds
            
            # Random burst logic: occasionally send a burst of requests with no delay
            if random.random() < 0.1:  # 10% chance of a burst
                print(f"Sending burst of requests at index {idx+1}")
                await asyncio.sleep(0)  # No delay between burst requests
            else:
                await asyncio.sleep(adjusted_interval)
    
    print("Asynchronous simulation completed.")

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Simulate continuous prediction requests with varying intervals.")
    parser.add_argument('--data_path', type=str, default="../datasets/fraudTrain.csv", help="Path to the training data CSV file.")
    parser.add_argument('--truncate_size', type=int, default=100000, help="Number of records to truncate the dataset to.")
    parser.add_argument('--endpoint', type=str, default=PREDICT_ENDPOINT, help="Prediction API endpoint.")
    parser.add_argument('--interval', type=int, default=1, help="Base time interval between requests in seconds.")
    parser.add_argument('--variability', type=float, default=0.5, help="Variability in the request interval (0 to 1).")
    parser.add_argument('--oversample', action='store_true', help="Whether to oversample fraudulent transactions.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    PREDICT_ENDPOINT = args.endpoint  # Update the endpoint if specified
    data = load_data(args.data_path, truncate_size=args.truncate_size, oversample_fraud=args.oversample)
    
    if not data.empty:
        asyncio.run(simulate_requests_async(data, base_interval=args.interval, variability=args.variability))
    else:
        print("No data available for simulation.")
