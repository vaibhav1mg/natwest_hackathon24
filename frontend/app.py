# frontend.py

import ast
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import requests
import os
import json
from pathlib import Path
from PIL import Image  # Correct import for image handling
from io import BytesIO
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from streamlit_option_menu import option_menu  # For the top navigation bar

# Define API endpoints
BASE_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{BASE_URL}/predict/"
METRICS_ENDPOINT = f"{BASE_URL}/metrics/"
MODEL_INFO_ENDPOINT = f"{BASE_URL}/model_info/"
DATA_DRIFT_ENDPOINT = f"{BASE_URL}/data_drift/"
LIST_MODELS_ENDPOINT = f"{BASE_URL}/list_models/"
ACTIVATE_MODEL_ENDPOINT = f"{BASE_URL}/activate_model/"
ROLLBACK_MODEL_ENDPOINT = f"{BASE_URL}/rollback/"
EXPLAIN_ENDPOINT = f"{BASE_URL}/explain/"
GENERATE_SHAP_PLOTS_ENDPOINT = f"{BASE_URL}/generate_shap_plots/"
SYSTEM_METRICS_ENDPOINT = f"{BASE_URL}/system_metrics/"
DETECT_DRIFT_ENDPOINT = f"{BASE_URL}/detect_drift/"
LOG_ALERT_ENDPOINT = f"{BASE_URL}/log_alert/"
TRANSACTION_DETAILS_ENDPOINT = f"{BASE_URL}/transaction/"
FILTERED_LOGS_ENDPOINT = f"{BASE_URL}/filtered_logs/"
TRAIN_MODEL_ENDPOINT = f"{BASE_URL}/train/"
MODEL_DETAIL_ENDPOINT = f"{BASE_URL}/model_details/"
SIMULATION_METRICS_ENDPOINT = f"{BASE_URL}/simulation_metrics/"
# Set page configuration
st.set_page_config(page_title="MLOps Platform Dashboard", layout="wide")

# Inject custom CSS for better aesthetics
def local_css():
    st.markdown("""
    <style>
    /* Top Navigation Bar */
    .navbar {
        background-color: #2F4F4F;
        padding: 10px;
    }
    .navbar a {
        color: white;
        padding: 14px 20px;
        text-decoration: none;
        font-size: 16px;
    }
    .navbar a:hover {
        background-color: #ddd;
        color: black;
    }
    /* Customize Streamlit elements */
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
    .stAlert {
        border-left: 5px solid #FF4B4B;
    }
    /* Adjust padding and margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Get the current script's directory
script_dir = Path(__file__).parent.resolve()

# Paths for assets and reports
assets_dir =  "../assets"
reports_dir = script_dir.parent / "reports"
logs_dir = "../logs"

# Top Navigation Bar using streamlit-option-menu
with st.container():
    selected = option_menu(
        menu_title=None,  # No title
        options=["Dashboard", "Guardrails", "Observability", "Explainability"],
        icons=["speedometer2", "shield-check", "bar-chart", "diagram-3"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#2F4F4F"},
            "icon": {"color": "white", "font-size": "16px"},
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "center"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )



# Function to get headers (authentication can be added here)
def get_headers():
    return {}

# Fetch active model information
def get_model_info():
    try:
        response = requests.get(MODEL_INFO_ENDPOINT, headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch model info.")
            return {}
    except Exception as e:
        st.error(f"Error fetching model info: {e}")
        return {}

# Fetch performance metrics
def get_metrics():
    try:
        response = requests.get(METRICS_ENDPOINT, headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch metrics.")
            return {}
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        return {}

# Caching the drift scores to avoid fetching them on every run
@st.cache_data(ttl=3600)  # Cache for 1 hour (adjust as needed)
def fetch_drift_scores():
    try:
        response = requests.get(DATA_DRIFT_ENDPOINT, headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch drift scores.")
            return {}
    except Exception as e:
        st.error(f"Error fetching drift scores: {e}")
        return {}

# Fetch list of available models
def fetch_available_models():
    try:
        response = requests.get(LIST_MODELS_ENDPOINT, headers=get_headers())
        if response.status_code == 200:
            return response.json().get("available_models", [])
        else:
            st.error("Failed to fetch available models.")
            return []
    except Exception as e:
        st.error(f"Error fetching available models: {e}")
        return []

# Activate a specific model
def activate_model(model_version):
    try:
        response = requests.post(f"{ACTIVATE_MODEL_ENDPOINT}{model_version}/", headers=get_headers())
        if response.status_code == 200:
            st.success(response.json().get("message", "Model activated successfully."))
        else:
            st.error(f"Failed to activate model: {response.text}")
    except Exception as e:
        st.error(f"Error activating model: {e}")

# Rollback to previous model
def rollback_model():
    try:
        response = requests.post(ROLLBACK_MODEL_ENDPOINT, headers=get_headers())
        if response.status_code == 200:
            st.success(response.json().get("message", "Rollback successful."))
        else:
            st.error(f"Failed to rollback model: {response.text}")
    except Exception as e:
        st.error(f"Error rolling back model: {e}")

# Explain a prediction (excluded as per instructions)
def explain_prediction(transaction_id):
    try:
        response = requests.get(f"{EXPLAIN_ENDPOINT}{transaction_id}/", headers=get_headers())
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error(f"Transaction ID {transaction_id} not found.")
            return {}
        else:
            st.error(f"Failed to explain prediction: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error explaining prediction: {e}")
        return {}

# Fetch and display transaction details
def display_transaction_details(transaction_id):
    try:
        response = requests.get(f"{TRANSACTION_DETAILS_ENDPOINT}{transaction_id}/", headers=get_headers())
        if response.status_code == 200:
            transaction = response.json()
            st.subheader(f"Transaction Details - ID: {transaction_id}")
            for key, value in transaction.items():
                st.write(f"**{key.capitalize()}**: {value}")

            # Optionally, display SHAP explanations if available
            # Since explainability is out of scope, we skip SHAP plots
        else:
            st.error(f"Failed to fetch details for Transaction ID {transaction_id}.")
    except Exception as e:
        st.error(f"Error fetching transaction details: {e}")

# Function to generate SHAP plots (excluded as per instructions)
def generate_shap_plots_api(model_name, feature_name):
    try:
        params = {
            "model_name": model_name,
            "dependence_feature": feature_name
        }
        response = requests.get(GENERATE_SHAP_PLOTS_ENDPOINT, params=params)
        if response.status_code == 200:
            res_json = response.json()
            if res_json.get("status") == "success":
                st.success(f"SHAP plots generated for model '{model_name}' and feature '{feature_name}'.")
                return res_json.get("summary_plot"), res_json.get("dependence_plot")
            else:
                st.error(res_json.get("message", "Failed to generate SHAP plots."))
                return None, None
        else:
            st.error("Failed to generate SHAP plots via the backend.")
            return None, None
    except Exception as e:
        st.error(f"Error generating SHAP plots: {e}")
        return None, None

# Enhanced Log Viewer with Filtering
def fetch_filtered_logs(start_date=None, end_date=None, transaction_id=None, anomaly=None, prediction=None, merchant=None):
    try:
        params = {}
        if start_date:
            params['start_date'] = start_date.isoformat()
        if end_date:
            params['end_date'] = end_date.isoformat()
        if transaction_id:
            try:
                params['transaction_id'] = int(transaction_id)
            except ValueError:
                st.error("Transaction ID must be an integer.")
                return []
        if anomaly and anomaly != "All":
            params['anomaly'] = True if anomaly == "True" else False
        if prediction and prediction != "All":
            try:
                params['prediction'] = int(prediction)
            except ValueError:
                st.error("Prediction outcome must be 0 or 1.")
                return []
        if merchant and merchant != "All":
            params['merchant'] = merchant

        response = requests.get(FILTERED_LOGS_ENDPOINT, params=params, headers=get_headers())
        if response.status_code == 200:
            return response.json().get("filtered_transactions", [])
        else:
            st.error("Failed to fetch filtered logs.")
            return []
    except Exception as e:
        st.error(f"Error fetching filtered logs: {e}")
        return []

# Function to fetch detailed model information
def fetch_model_details():
    try:
        response = requests.get(MODEL_DETAIL_ENDPOINT, headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch model details.")
            return []
    except Exception as e:
        st.error(f"Error fetching model details: {e}")
        return []



# Initialize session state for example values
if 'example_selected' not in st.session_state:
    st.session_state.example_selected = None

# Define example data
example_1 = {
    "merchant": "fraud_Rutherford-Mertz",
    "category": "grocery_pos",
    "amt": 281.06,
    "gender": "Male",
    "job": "Soil scientist",
    "lat": 35.9946,
    "long": -81.7266,
    "city_pop": 885,
    "merch_lat": 36.430124,
    "merch_long": -81.179483,
    "unix_time": 1325466397
}

example_2 = {
    "merchant": "fraud_Johnston-Casper",
    "category": "travel",
    "amt": 3.19,
    "gender": "Male",
    "job": "Furniture designer",
    "lat": 44.2529,
    "long": -85.017,
    "city_pop": 1126,
    "merch_lat": 44.959148,
    "merch_long": -85.884734,
    "unix_time": 1371816917
}

# Map the examples to the selection
examples = {
    "None": None,
    "Example 1": example_1,
    "Example 2": example_2
}


# Initialize session state if it doesn't exist
if 'alerts_list' not in st.session_state:
    st.session_state.alerts_list = []

if 'alert_services' not in st.session_state:
    st.session_state.alert_services = {
        "email": {"smtp": "mail.privateemail.com:465", "username": "", "password": "", "enabled": True, "active": True},
        "telegram": {"api_key": "", "enabled": False, "active": True},
        "discord": {"api_key": "", "enabled": True, "active": True},
    }


# Main Content Sections

# Dashboard Section
if selected == "Dashboard":
    st.title("📊 Central Dashboard")
    model_info = get_model_info()
    metrics = get_metrics()

    if model_info and metrics:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
            col1.metric("Model Version", model_info.get("model_version", "N/A").split('_', 1)[-1])
            col2.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.2f}%")
            col3.metric("Precision", f"{metrics.get('precision', 0.0) * 100:.2f}%")
            col4.metric("Recall", f"{metrics.get('recall', 0.0) * 100:.2f}%")
            col5.metric("ROC AUC", f"{metrics.get('roc_auc', 0.0):.2f}")

        st.markdown("---")

        # Observability Metrics Overview with auto-refresh (real-time update)
        st.subheader("🔍 Observability Metrics Overview")

        # Placeholder for observability metrics
        obs_metrics_placeholder = st.empty()

        # To keep track of line chart data (for continuous updates)
        line_metrics_history = {"time": [], "Avg Response Time (ms)": [], "Throughput (req/sec)": []}

        # Function to update observability metrics every 2 seconds
        def update_observability_metrics():
            global line_metrics_history  # Track historical data

            try:
                response = requests.get(SIMULATION_METRICS_ENDPOINT, headers=get_headers())
                if response.status_code == 200:
                    sim_metrics = response.json()

                    # Add time-based data for continuous line updates
                    current_time = time.strftime('%H:%M:%S')
                    line_metrics_history["time"].append(current_time)

                    # Convert Avg Response Time to milliseconds and add to history
                    avg_response_time_ms = sim_metrics.get("average_response_time", 0.0) * 1000
                    line_metrics_history["Avg Response Time (ms)"].append(avg_response_time_ms)
                    line_metrics_history["Throughput (req/sec)"].append(sim_metrics.get("throughput", 0.0))

                    # Limit history to the last 10 points
                    if len(line_metrics_history["time"]) > 10:
                        for key in line_metrics_history:
                            line_metrics_history[key] = line_metrics_history[key][-10:]

                    # Determine dynamic range for y-axis with buffers
                    def calculate_buffered_range(metric_values):
                        min_val = min(metric_values)
                        max_val = max(metric_values)
                        range_span = max_val - min_val if max_val - min_val !=0 else 1
                        buffer_lower = range_span * 0.10  # 10% lower buffer
                        buffer_upper = range_span * 0.50  # 50% upper buffer
                        return [max(min_val - buffer_lower, 0), max_val + buffer_upper]

                    # Calculate the range for throughput and response time with buffer
                    throughput_range = calculate_buffered_range(line_metrics_history["Throughput (req/sec)"])
                    response_time_range = calculate_buffered_range(line_metrics_history["Avg Response Time (ms)"])

                    # Create a figure with two subplots
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Avg Response Time (ms)", "Throughput (req/sec)"))

                    # Add a filled area trace for Avg Response Time in ms
                    fig.add_trace(go.Scatter(
                        x=line_metrics_history["time"],
                        y=line_metrics_history["Avg Response Time (ms)"],
                        fill='tozeroy',  # Fill to the bottom
                        mode='lines+markers',
                        name="Avg Response Time (ms)",
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=8)
                    ), row=1, col=1)

                    # Add a filled area trace for throughput
                    fig.add_trace(go.Scatter(
                        x=line_metrics_history["time"],
                        y=line_metrics_history["Throughput (req/sec)"],
                        fill='tozeroy',  # Fill to the bottom
                        mode='lines+markers',
                        name="Throughput (req/sec)",
                        line=dict(color='#ff7f0e', width=2),
                        marker=dict(size=8)
                    ), row=1, col=2)

                    # Update layout for the subplots
                    fig.update_layout(
                        height=400,
                        template="plotly_white",
                        showlegend=False
                    )

                    # Update individual y-axes with dynamic range and buffers
                    fig.update_yaxes(title_text="Avg Response Time (ms)", range=response_time_range, row=1, col=1)
                    fig.update_yaxes(title_text="Throughput (req/sec)", range=throughput_range, row=1, col=2)
                    fig.update_xaxes(title_text="Time", row=1, col=1)
                    fig.update_xaxes(title_text="Time", row=1, col=2)

                    # Update the observability metrics section with the updated figure
                    obs_metrics_placeholder.plotly_chart(fig, use_container_width=True)

                else:
                    obs_metrics_placeholder.error("Failed to fetch simulation metrics.")

            except Exception as e:
                obs_metrics_placeholder.error(f"Error fetching simulation metrics: {e}")

        # System Health Metrics with auto-refresh (real-time update)
        st.subheader("🖥️ System Health Metrics")

        # Placeholder to update system health metrics
        sys_metrics_placeholder = st.empty()

        # Function to update system health metrics every 2 seconds
        def update_system_health_metrics():
            try:
                response = requests.get(SYSTEM_METRICS_ENDPOINT, headers=get_headers())
                if response.status_code == 200:
                    sys_metrics = response.json()

                    # Create Gauge charts for CPU, Memory, and Disk Usage
                    fig_sys = make_subplots(rows=1, cols=3,
                                            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
                                            subplot_titles=("CPU Usage", "Memory Usage", "Disk Usage"))

                    fig_sys.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=sys_metrics.get("cpu_usage_percent", 0.0),
                        title={'text': "CPU Usage (%)"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "#1f77b4"},
                               'steps': [
                                   {'range': [0, 50], 'color': "#a6cee3"},
                                   {'range': [50, 100], 'color': "#1f78b4"}]},
                    ), row=1, col=1)

                    fig_sys.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=sys_metrics.get("memory_usage_percent", 0.0),
                        title={'text': "Memory Usage (%)"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "#2ca02c"},
                               'steps': [
                                   {'range': [0, 50], 'color': "#98df8a"},
                                   {'range': [50, 100], 'color': "#2ca02c"}]},
                    ), row=1, col=2)

                    fig_sys.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=sys_metrics.get("disk_usage_percent", 0.0),
                        title={'text': "Disk Usage (%)"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "#d62728"},
                               'steps': [
                                   {'range': [0, 50], 'color': "#ff9896"},
                                   {'range': [50, 100], 'color': "#d62728"}]},
                    ), row=1, col=3)

                    fig_sys.update_layout(height=360, template="plotly_white")

                    # Update the system health metrics section
                    sys_metrics_placeholder.plotly_chart(fig_sys, use_container_width=True)

                else:
                    sys_metrics_placeholder.error("Failed to fetch system metrics.")

            except Exception as e:
                sys_metrics_placeholder.error(f"Error fetching system metrics: {e}")

        st.markdown("---")

        # Active Models Table
        st.subheader("🛠️ Active Models")
        model_details = fetch_model_details()

        
                        
        if model_details:
            models_df = pd.DataFrame(model_details)
            # Select relevant columns to display
            display_columns = ["model_version", "accuracy", "precision", "recall", "f1_score", "roc_auc", "timestamp"]
            models_display_df = models_df[display_columns]

            # Display the table with full width
            st.dataframe(models_display_df.style.format({
                "accuracy": "{:.2%}",
                "precision": "{:.2%}",
                "recall": "{:.2%}",
                "f1_score": "{:.2%}",
                "roc_auc": "{:.2f}"
            }).set_properties(**{'background-color': '#f0f2f6', 'color': '#000000'}), use_container_width=True)

            # Enable model selection
            selected_model = st.selectbox("Select a Model to View Hyperparameters", models_df['model_version'].tolist())

            # Button to view hyperparameters in a dialog
            if st.button("🔍 View Hyperparameters"):
                selected_details = models_df[models_df['model_version'] == selected_model].iloc[0]

                # Open a dialog box to show hyperparameters
                @st.dialog("🧰 Hyperparameters", width="medium")
                def view_hyperparameters():
                    st.write("### Model Hyperparameters")
                    try:
                        # Convert the string to a Python dictionary using ast.literal_eval
                        hyperparams = ast.literal_eval(selected_details['hyperparameters'])

                        # Pretty print the dictionary as JSON
                        st.json(hyperparams, expanded=True)  # Pretty print the JSON

                    except (ValueError, SyntaxError):
                        # Fallback: Display as raw text if conversion fails
                        st.text("Unable to parse hyperparameters as JSON.")
                        st.text(selected_details['hyperparameters'])
                
                view_hyperparameters()
        else:
            st.write("No models available.")


        st.markdown("---")
        # Recent Transactions with Clickable Rows
        st.subheader("📝 Recent Transactions")
        try:
            log_path = logs_dir / "transactions.log"
            if log_path.exists():
                with open(log_path, "r") as log_file:
                    logs = log_file.readlines()
                # Parse JSON logs, skip invalid lines
                transactions = []
                for log in logs[-10:]:  # Get last 10 transactions
                    try:
                        transaction = json.loads(log.strip())
                        transactions.append(transaction)
                    except json.JSONDecodeError:
                        continue
                if transactions:
                    transactions_df = pd.DataFrame(transactions)
                    # Drop columns that may not be needed for display
                    columns_to_display = ["timestamp", "transaction_id", "merchant", "amt", "prediction", "probability_of_fraud", "anomaly"]
                    transactions_df = transactions_df[columns_to_display]

                    # Display transactions with a selectbox to choose a transaction to view details
                    transaction_ids = transactions_df['transaction_id'].tolist()
                    selected_id = st.selectbox("Select a Transaction ID to View Details", transaction_ids)
                    if st.button("🔍 View Details"):
                        display_transaction_details(selected_id)
                else:
                    st.write("No valid transaction logs found.")
            else:
                st.warning("Transaction log file not found.")
        except Exception as e:
            st.error(f"Error loading transactions: {e}")

        # Continuously update system health metrics every 2 seconds
        placeholder = st.empty()
        while True:
            update_system_health_metrics()
            update_observability_metrics()
            time.sleep(2)  # Refresh every 2 seconds

# Guardrails Section
elif selected == "Guardrails":
    st.title("🛡️ Guardrails")

    # Fetch metrics within Guardrails section
    metrics = get_metrics()
    model_info = get_model_info()  # Ensure model_info is available

    # Data Drift Visualization
    st.subheader("📈 Data Drift Detection")

    # Fetch cached drift scores
    drift_scores = fetch_drift_scores()

    if drift_scores:
        # Remove 'unix_time' from the drift scores
        filtered_drift_scores = {k: v for k, v in drift_scores.items() if k != 'unix_time'}

        # Convert to DataFrame for easy handling
        drift_df = pd.DataFrame({
            "Feature": list(filtered_drift_scores.keys()),
            "Statistic": [
                filtered_drift_scores[f]["ks_statistic"] if "ks_statistic" in filtered_drift_scores[f] 
                else filtered_drift_scores[f]["chi2_statistic"] 
                for f in filtered_drift_scores
            ],
            "P-Value": [filtered_drift_scores[f]["p_value"] for f in filtered_drift_scores]
        })

        # Filter for significant drifts (P-Value < 0.05)
        significant_drift_df = drift_df[drift_df["P-Value"] < 0.05]

        # Visualize significant drifts with log scale for better clarity
        if not significant_drift_df.empty:
            # Create two columns: left for chart, right for p-values
            col1, col2 = st.columns([4, 1])  # Adjust column width ratio (3:1)

            # Left column: Bar chart with drift statistics
            with col1:
                fig = px.bar(
                    significant_drift_df,
                    x="Feature",
                    y="Statistic",
                    title="Significant Feature Drift Scores (Log Scale)",
                    text_auto=True,
                    color="Statistic",
                    color_continuous_scale='Viridis',
                    log_y=True  # Use log scale for y-axis
                )
                fig.update_layout(
                    title_font_size=24,
                    title_y=0.96,  # Adjust this value to control the vertical position of the title
                    xaxis_title="Feature",
                    yaxis_title="Drift Statistic (Log Scale)",
                    font=dict(size=16),
                    margin=dict(t=40)  # Keep a margin above the title (default margin)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Right column: P-values in a smaller table
            with col2:
                st.subheader("P-Values")
                st.dataframe(significant_drift_df[["Feature", "P-Value"]].reset_index(drop=True))
        else:
            st.write("No significant drift detected in the features.")
    else:
        st.write("No drift scores available.")

    st.markdown("---")

    # Performance Alerts
    st.subheader("🚨 Performance Alerts")
    # Define thresholds
    ACCURACY_THRESHOLD = 0.95
    PRECISION_THRESHOLD = 0.70
    RECALL_THRESHOLD = 0.75

    if metrics:
        accuracy = metrics.get('accuracy', 1.0)
        precision = metrics.get('precision', 1.0)
        recall = metrics.get('recall', 1.0)

        alerts = []
        if accuracy < ACCURACY_THRESHOLD:
            alerts.append(f"Model accuracy has dropped below {ACCURACY_THRESHOLD*100}%. Current: {accuracy*100:.2f}%")
        if precision < PRECISION_THRESHOLD:
            alerts.append(f"Model precision has dropped below {PRECISION_THRESHOLD*100}%. Current: {precision*100:.2f}%")
        if recall < RECALL_THRESHOLD:
            alerts.append(f"Model recall has dropped below {RECALL_THRESHOLD*100}%. Current: {recall*100:.2f}%")

        if alerts:
            for alert in alerts:
                st.error(f"❗️ **Alert:** {alert}")
        else:
            st.success("✅ All performance metrics are within acceptable thresholds.")
    else:
        st.warning("Performance metrics not available.")


    st.markdown("---")

    # Fetch model details and available models from your backend
    model_details = fetch_model_details()
    model_info = get_model_info()  # Ensure model_info is available
    available_models = fetch_available_models()

    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])

    # Column 1: Train New Model
    with col1:
        st.subheader("🛠️ Train New Model")

        if model_details:

            # Function to handle model training dialog
            @st.dialog("Train a New Model", width="large")
            def train_model_dialog():
                st.write("### Set Hyperparameters")

                # Input placeholders for hyperparameters
                n_estimators = st.number_input("n_estimators", min_value=50, max_value=500, value=100, step=10)
                max_depth = st.number_input("max_depth", min_value=5, max_value=50, value=None, step=1)
                min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=20, value=2, step=1)
                min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=20, value=1, step=1)
                random_state = st.number_input("random_state", min_value=0, max_value=100, value=42, step=1)

                st.write("You can modify the above hyperparameters or leave them at their default values.")

                # Function to send POST request to the FastAPI endpoint for training
                def train_model_api_call():
                    # Prepare the payload with the hyperparameters
                    payload = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "random_state": random_state
                    }
                    try:
                        # Send POST request to FastAPI backend
                        response = requests.post(TRAIN_MODEL_ENDPOINT, params=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Training Complete! Metrics: {result['metrics']}")
                        else:
                            st.error(f"Error: {response.status_code}, {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error: {e}")

                # Create two columns for buttons
                col1, col2 = st.columns(2)

                # Place buttons in two separate columns (same row)
                with col1:
                    if st.button("🚀 Train Model"):
                        # Display feedback directly inside the dialog
                        st.info("Initializing training...")
                        train_model_api_call()  # Call FastAPI to train the model

                with col2:
                    if st.button("🎯 Train with Optuna"):
                        # Display feedback directly inside the dialog
                        st.info("Starting Optuna optimization...")
                        time.sleep(1)
                        st.info("Optuna optimization in progress...")
                        time.sleep(6)

            # Button to open the training dialog
            if st.button("⚙️ Train New Model"):
                train_model_dialog()  # Open the dialog for training the model
                
        else:
            st.warning("No models available to train.")


    # Column 2: Model Version Control
    with col2:
        st.subheader("🔄 Model Version Control")

        # Display current active model
        current_version = model_info.get("model_version", "N/A") if model_info else "N/A"
        st.write(f"**Current Active Model:** {current_version}")

        if available_models:
            # Dropdown for model selection
            selected_model = st.selectbox("Select Model Version to Activate", available_models, index=0)
            
            # Activate model button
            if st.button("✅ Activate Selected Model"):
                activate_model(selected_model)
                st.toast(f"Model version {selected_model} activated successfully! ✅", icon="✅")
        else:
            st.warning("No available models to activate.")

        # Rollback button
        if st.button("🔙 Rollback to Previous Version"):
            rollback_model()
            st.toast("Model rolled back to the previous version successfully! 🔙", icon="🔙")


    # Separator
    st.markdown("---")

    # 1. Add Alerts Section
    st.subheader("⚠️ Alert Configuration")




    # Button to add alerts
    if st.button("➕ Add Alerts", help="Configure a new alert"):
        # Open dialog for alert configuration
        @st.dialog("🛠️ Configure New Alert", width="medium")
        def alert_configuration_dialog():
            st.write("### Add New Alert")

            # 'If' field: Select sector (e.g., Accuracy, CPU usage)
            sector = st.selectbox("Sector", ["Accuracy", "CPU Usage", "Memory Usage", "SSD Usage", "Fraud Transaction"])

            # 'When' field: Select a threshold condition
            condition = st.selectbox("Condition", ["Greater than", "Less than", "Equal to"])
            threshold_value = st.number_input("Threshold Value", min_value=0.0, max_value=100.0, step=1.0)

            # 'Notify via' field: Checkbox options for notification methods
            st.write("### Notify via")
            notify_email = st.checkbox("Email")
            notify_telegram = st.checkbox("Telegram")
            notify_discord = st.checkbox("Discord")

            # Add Alert button
            if st.button("Add Alert"):
                new_alert = {
                    "sector": sector,
                    "condition": condition,
                    "threshold": threshold_value,
                    "notify_email": notify_email,
                    "notify_telegram": notify_telegram,
                    "notify_discord": notify_discord,
                }
                st.session_state.alerts_list.append(new_alert)  # Store in session state
                st.success(f"Alert for {sector} added successfully!")
                st.rerun()  # Close the dialog after adding alert

        alert_configuration_dialog()

    # Display all added alerts
    if st.session_state.alerts_list:
        st.write("### Active Alerts")
        for idx, alert in enumerate(st.session_state.alerts_list):
            alert_str = (
                f"**{alert['sector']}** {alert['condition']} {alert['threshold']} | "
                f"**Notify via**: Email: {'✅' if alert['notify_email'] else '❌'}, "
                f"Telegram: {'✅' if alert['notify_telegram'] else '❌'}, "
                f"Discord: {'✅' if alert['notify_discord'] else '❌'}"
            )
            col1, col2 = st.columns([10, 1])
            col1.markdown(f"🔔 {alert_str}")
            if col2.button("❌", key=f"remove_{idx}"):
                st.session_state.alerts_list.pop(idx)  # Remove from session state
                st.rerun()

    # Separator
    st.markdown("---")

    # Create a row with 3 equal parts (1:3 ratio for each setting)
    col1, col2, col3 = st.columns(3)

    # Email Settings with SMTP and username/password fields
    with col1:
        st.write("### 📧 Email SMTP Configuration")
        email_smtp = st.text_input("SMTP:Port", value=st.session_state.alert_services['email']['smtp'], key="email_smtp", help="Enter SMTP server and port")
        email_username = st.text_input("Username", value=st.session_state.alert_services['email']['username'], key="email_username", help="Enter email username")
        email_password = st.text_input("Password", value=st.session_state.alert_services['email']['password'], type="password", key="email_password", help="Enter email password")
        email_enabled = st.checkbox("Enabled", value=st.session_state.alert_services['email']['enabled'], key="email_enabled")
        st.session_state.alert_services['email']['smtp'] = email_smtp
        st.session_state.alert_services['email']['username'] = email_username
        st.session_state.alert_services['email']['password'] = email_password
        st.session_state.alert_services['email']['enabled'] = email_enabled
        # st.markdown(f"**Status:** {'🟢 Active' if email_enabled else '🔴 Inactive'}")

    # Telegram Settings with API key
    with col2:
        st.write("###  Telegram Configuration")
        telegram_api_key = st.text_input("Telegram Token", value=st.session_state.alert_services['telegram']['api_key'], key="telegram_api_key", type="password", help="Enter your Telegram bot API key")
        telegram_enabled = st.checkbox("Enabled", value=st.session_state.alert_services['telegram']['enabled'], key="telegram_enabled")
        st.session_state.alert_services['telegram']['api_key'] = telegram_api_key
        st.session_state.alert_services['telegram']['enabled'] = telegram_enabled
        # st.markdown(f"**Status:** {'🟢 Active' if telegram_enabled else '🔴 Inactive'}")

    # Discord Settings with API key
    with col3:
        st.write("### Discord Configuration")
        discord_api_key = st.text_input("Discord Token", value=st.session_state.alert_services['discord']['api_key'], key="discord_api_key", type="password", help="Enter your Discord bot API key")
        discord_enabled = st.checkbox("Enabled", value=st.session_state.alert_services['discord']['enabled'], key="discord_enabled")
        st.session_state.alert_services['discord']['api_key'] = discord_api_key
        st.session_state.alert_services['discord']['enabled'] = discord_enabled
        # st.markdown(f"**Status:** {'🟢 Active' if discord_enabled else '🔴 Inactive'}")

        
    # Separator
    st.markdown("---")

    # Latest Alerts Section (Placeholder)
    st.subheader("📜 Latest Alerts")
    # Mock alerts log
    latest_alerts_mock = [
        {"labels": {"alertname": "High CPU Usage", "severity": "critical"}, "annotations": {"description": "CPU Usage above 90%"}, "startsAt": "2024-10-09T12:34:56Z"},
    ]
    if latest_alerts_mock:
        for alert in latest_alerts_mock:
            alert_display = (
                f"**Alert Name:** {alert['labels'].get('alertname', 'N/A')} | "
                f"**Severity:** {alert['labels'].get('severity', 'N/A')} | "
                f"**Description:** {alert['annotations'].get('description', 'N/A')} | "
                f"**Timestamp:** {alert['startsAt']}"
            )
            st.markdown(f":warning: {alert_display}")
    else:
        st.write("No alerts found.")

        
    st.markdown("---")


# Observability Section
elif selected == "Observability":
    st.title("🔭 Observability")

    # Real-time Metrics
    st.subheader("📈 Real-time Metrics")
    try:
        response = requests.get(SIMULATION_METRICS_ENDPOINT, headers=get_headers())
        if response.status_code == 200:
            sim_metrics = response.json()
            with st.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Requests", sim_metrics.get("total_requests", 0))
                col2.metric("Successful Predictions", sim_metrics.get("successful_predictions", 0))
                col3.metric("Failed Predictions", sim_metrics.get("failed_predictions", 0))
                col4.metric("Avg Response Time (ms)", f"{sim_metrics.get('average_response_time', 0.0):.4f}")
                col5.metric("Throughput (req/sec)", f"{sim_metrics.get('throughput', 0.0):.2f}")
        else:
            st.error("Failed to fetch simulation metrics.")
    except Exception as e:
        st.error(f"Error fetching simulation metrics: {e}")

    st.markdown("---")

    # System Health Metrics
    st.subheader("🖥️ System Health Metrics")
    try:
        response = requests.get(SYSTEM_METRICS_ENDPOINT, headers=get_headers())
        if response.status_code == 200:
            sys_metrics = response.json()
            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.metric("CPU Usage (%)", sys_metrics.get("cpu_usage_percent", 0.0))
                col2.metric("Memory Usage (%)", sys_metrics.get("memory_usage_percent", 0.0))
                col3.metric("Disk Usage (%)", sys_metrics.get("disk_usage_percent", 0.0))
        else:
            st.error("Failed to fetch system metrics.")
    except Exception as e:
        st.error(f"Error fetching system metrics: {e}")

    st.markdown("---")

    # Anomaly Alerts (now Fraud Alerts)
    st.subheader("⚠️ Fraud Alerts")

    # Assuming logs_dir is already defined
    try:
        log_path = logs_dir / "transactions.log"
        if log_path.exists():
            with open(log_path, "r") as log_file:
                logs = log_file.readlines()
            
            # Parse JSON logs and filter fraud transactions
            fraud_transactions = []
            for log in logs:
                try:
                    transaction = json.loads(log.strip())
                    # Assuming `prediction` is a boolean indicating fraud (True = Fraud)
                    if transaction.get("prediction") == True:
                        fraud_transactions.append(transaction)
                except json.JSONDecodeError:
                    continue

            if fraud_transactions:
                fraud_df = pd.DataFrame(fraud_transactions)
                # Display relevant columns
                columns_to_display = ["timestamp", "transaction_id", "merchant", "amt", "prediction", "probability_of_fraud"]
                fraud_df = fraud_df[columns_to_display]
                # Display the dataframe with full width and custom styling
                st.dataframe(fraud_df.style.set_properties(**{'background-color': '#f0f2f6', 'color': '#000000'}), use_container_width=True)
            else:
                st.write("No fraud detected.")
        else:
            st.warning("Transaction log file not found.")
    except Exception as e:
        st.error(f"Error loading fraud transactions: {e}")

    st.markdown("---")

    # Detailed Transaction Views
    st.subheader("🔍 Detailed Transaction Views")

    try:
        log_path = logs_dir / "transactions.log"
        if log_path.exists():
            with open(log_path, "r") as log_file:
                logs = log_file.readlines()
            # Parse JSON logs, display as a selectable DataFrame
            transactions = []
            for log in logs[-50:]:  # Display last 50 transactions
                try:
                    transaction = json.loads(log.strip())
                    transactions.append(transaction)
                except json.JSONDecodeError:
                    continue
            if transactions:
                transactions_df = pd.DataFrame(transactions)
                transactions_df = transactions_df.sort_values(by="timestamp", ascending=False)
                # Display a selectable table
                transaction_ids = transactions_df['transaction_id'].tolist()
                selected_id = st.selectbox("Select a Transaction ID to View Details", transaction_ids)
                if st.button("🔍 View Details"):
                    display_transaction_details(selected_id)
            else:
                st.write("No valid transaction logs found.")
        else:
            st.warning("Transaction log file not found.")
    except Exception as e:
        st.error(f"Error loading transactions: {e}")

    st.markdown("---")

    # Enhanced Log Viewer with Filtering
    st.subheader("🔎 Enhanced Log Viewer with Filtering")

    with st.expander("📂 Filter Logs"):
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.today())
        with col2:
            end_date = st.date_input("End Date", value=datetime.today())

        transaction_id = st.text_input("Transaction ID (Optional)")
        anomaly = st.selectbox("Anomaly Status (Optional)", ["All", "True", "False"])
        prediction = st.selectbox("Prediction Outcome (Optional)", ["All", "0", "1"])
        merchant = st.text_input("Merchant Name (Optional)")

        if st.button("🔍 Apply Filters"):
            filtered_transactions = fetch_filtered_logs(
                start_date=start_date,
                end_date=end_date,
                transaction_id=transaction_id if transaction_id else None,
                anomaly=anomaly if anomaly != "All" else None,
                prediction=prediction if prediction != "All" else None,
                merchant=merchant if merchant else None
            )
            if filtered_transactions:
                logs_df = pd.DataFrame(filtered_transactions)
                st.dataframe(logs_df.style.set_properties(**{'background-color': '#f0f2f6', 'color': '#000000'}))
            else:
                st.write("No transactions match the filter criteria.")

    st.markdown("---")

    # Search Functionality
    st.subheader("🔍 Search Transactions")
    search_query = st.text_input("Enter Transaction ID or Merchant Name to Search")
    if st.button("🔎 Search"):
        try:
            # Attempt to search by transaction_id first
            try:
                transaction_id = int(search_query)
                response = requests.get(f"{TRANSACTION_DETAILS_ENDPOINT}{transaction_id}/", headers=get_headers())
                if response.status_code == 200:
                    transaction = response.json()
                    st.write("### 📄 Transaction Details")
                    for key, value in transaction.items():
                        st.write(f"**{key.capitalize()}**: {value}")
                else:
                    st.error(f"Transaction ID {transaction_id} not found.")
            except ValueError:
                # If not an integer, search by merchant name
                filtered_logs = fetch_filtered_logs(merchant=search_query)
                if filtered_logs:
                    logs_df = pd.DataFrame(filtered_logs)
                    st.dataframe(logs_df.style.set_properties(**{'background-color': '#f0f2f6', 'color': '#000000'}))
                else:
                    st.write("No transactions match the merchant name.")
        except Exception as e:
            st.error(f"Error during search: {e}")

    st.markdown("---")

    # Log Viewer
    st.subheader("📜 Transaction Logs")
    try:
        log_path = logs_dir / "transactions.log"
        if log_path.exists():
            with open(log_path, "r") as log_file:
                logs = log_file.readlines()
            # Display the last 20 log entries
            last_logs = logs[-20:]
            for log in last_logs:
                try:
                    transaction = json.loads(log.strip())
                    log_display = (
                        f"**Timestamp:** {transaction.get('timestamp', 'N/A')} | "
                        f"**ID:** {transaction.get('transaction_id', 'N/A')} | "
                        f"**Merchant:** {transaction.get('merchant', 'N/A')} | "
                        f"**Amount:** {transaction.get('amt', 'N/A')} | "
                        f"**Prediction:** {transaction.get('prediction', 'N/A')} | "
                        f"**Prob. Fraud:** {transaction.get('probability_of_fraud', 'N/A')} | "
                        f"**Anomaly:** {transaction.get('anomaly', 'N/A')}"
                    )
                    st.markdown(f"• {log_display}")
                except json.JSONDecodeError:
                    st.write(log.strip())
        else:
            st.warning("Transaction log file not found.")
    except Exception as e:
        st.error(f"Error loading logs: {e}")

# Explainability Section (Existing SHAP-related code retained)
elif selected == "Explainability":
    st.title("🧠 Explainability")

    # Fetch available models
    available_models = fetch_available_models()
    if not available_models:
        st.warning("No models available.")

    # Dropdown to select model, default to the latest model
    latest_model = available_models[-1] if available_models else None
    selected_model = st.selectbox("Select Model", options=available_models, index=len(available_models)-1 if available_models else 0)

    # Dropdown to select feature, default to 'amt'
    def fetch_feature_names(model_name):
        try:
            # Assuming all models use the same dataset structure
            test_data_path = os.path.join("../datasets/", "fraudTest.csv")
            if not os.path.exists(test_data_path):
                st.error(f"Dataset file not found at {test_data_path}.")
                return []

            test_data = pd.read_csv(test_data_path).drop(columns=["is_fraud"])
            # Preprocessing (mirror backend)
            columns_to_drop = [
                'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last',
                'street', 'city', 'state', 'zip', 'dob', 'trans_num'
            ]
            test_data = test_data.drop(columns=columns_to_drop, errors='ignore')
            categorical_columns = ['merchant', 'category', 'gender', 'job']
            for col in categorical_columns:
                test_data[col] = test_data[col].astype('category').cat.codes  # Convert categories to numeric codes

            feature_names = list(test_data.columns)
            return feature_names
        except Exception as e:
            st.error(f"Error fetching feature names: {e}")
            return []

    feature_options = fetch_feature_names(selected_model)
    if not feature_options:
        st.warning("No features available for the selected model.")

    default_feature = "amt" if "amt" in feature_options else feature_options[0] if feature_options else None
    selected_feature = st.selectbox("Select Feature for Dependence Plot", options=feature_options, index=feature_options.index(default_feature) if default_feature else 0)

    # Button to generate and display SHAP plots
    if st.button("📈 Generate SHAP Plots"):
        if selected_feature:
            # Define model-specific assets directory
            model_assets_dir = assets_dir / selected_model
            summary_plot_path = model_assets_dir / "shap_summary.png"
            dependence_plot_path = model_assets_dir / f"shap_dependence_{selected_feature}.png"

            # Check if plots already exist
            plots_exist = summary_plot_path.exists() and dependence_plot_path.exists()

            if plots_exist:
                st.success("SHAP plots already exist. Displaying them...")
            else:
                st.info("Generating SHAP plots...")
                shap_summary_url, shap_dependence_url = generate_shap_plots_api(selected_model, selected_feature)
                if shap_summary_url and shap_dependence_url:
                    # Plots have been generated; proceed to display
                    pass
                else:
                    st.error("Failed to generate SHAP plots.")

            # Display SHAP Summary Plot
            if summary_plot_path.exists():
                st.subheader("📊 SHAP Summary Plot")
                try:
                    summary_img = Image.open(summary_plot_path)
                    st.image(summary_img, caption="SHAP Summary Plot", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading SHAP Summary Plot image: {e}")
            else:
                st.warning("SHAP summary plot not found.")

            # Display SHAP Dependence Plot
            if dependence_plot_path.exists():
                st.subheader(f"📉 SHAP Dependence Plot for '{selected_feature}'")
                try:
                    dependence_img = Image.open(dependence_plot_path)
                    st.image(dependence_img, caption=f"SHAP Dependence Plot for '{selected_feature}'", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading SHAP Dependence Plot image: {e}")
            else:
                st.warning(f"SHAP dependence plot for '{selected_feature}' not found.")
        else:
            st.error("Please select a valid feature to generate SHAP plots.")

        
        # Prediction Input Form
    st.markdown("---")
    st.subheader("🔮 Make a Prediction")
    with st.form(key='prediction_form'):
        # Example selection inside the form
        selected_example = st.selectbox(
            "Select Example to Pre-fill",
            options=["None", "Example 1", "Example 2"]
        )

        if selected_example != "None":
            # Update session state with selected example values
            st.session_state.example_selected = examples[selected_example]

        # Set default values based on the selected example (if any)
        example_values = st.session_state.example_selected or {}

        col1, col2 = st.columns(2)
        with col1:
            merchant = st.text_input("Merchant", value=example_values.get("merchant", ""))
            category = st.text_input("Category", value=example_values.get("category", ""))
            amt = st.number_input("Amount", min_value=0.0, step=0.01, value=example_values.get("amt", 0.0))
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if example_values.get("gender", "Male") == "Male" else 1)
            job = st.text_input("Job", value=example_values.get("job", ""))
        with col2:
            lat = st.number_input("Latitude", format="%.6f", value=example_values.get("lat", 0.0))
            long = st.number_input("Longitude", format="%.6f", value=example_values.get("long", 0.0))
            city_pop = st.number_input("City Population", min_value=0, value=example_values.get("city_pop", 0))
            merch_lat = st.number_input("Merchant Latitude", format="%.6f", value=example_values.get("merch_lat", 0.0))
            merch_long = st.number_input("Merchant Longitude", format="%.6f", value=example_values.get("merch_long", 0.0))
            unix_time = st.number_input("Unix Time", min_value=0, step=1, value=example_values.get("unix_time", 0))

        submit_button = st.form_submit_button(label='🔍 Predict')

    # Handle Prediction
    if submit_button:
        try:
            transaction_data = {
                "merchant": merchant,
                "category": category,
                "amt": amt,
                "gender": gender,
                "job": job,
                "lat": lat,
                "long": long,
                "city_pop": city_pop,
                "merch_lat": merch_lat,
                "merch_long": merch_long,
                "unix_time": int(unix_time)
            }
            response = requests.post(PREDICT_ENDPOINT, json=transaction_data)
            if response.status_code == 200:
                prediction_result = response.json()
                st.success(f"**Prediction:** {'Fraudulent' if prediction_result['prediction'] == 1 else 'Legitimate'}")
                st.info(f"**Probability of Fraud:** {prediction_result['probability_of_fraud']:.4f}")
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    