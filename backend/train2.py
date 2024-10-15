import os
import numpy as np
import pandas as pd
import dill
import datetime
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Paths
MODEL_DIR = "../models/"
METRICS_DIR = "../metrics/"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(METRICS_DIR):
    os.makedirs(METRICS_DIR)

def train_model(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
    model_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # %% 1. Read and Truncate Data
    TRAIN_DATA_PATH = "../datasets/fraudTrain.csv"
    TEST_DATA_PATH = "../datasets/fraudTest.csv"
    
    train_data = pd.read_csv(TRAIN_DATA_PATH).sample(n=30000, random_state=42).reset_index(drop=True)
    test_data = pd.read_csv(TEST_DATA_PATH).sample(n=5000, random_state=42).reset_index(drop=True)

    # Track dataset using DVC
    subprocess.run(["dvc", "add", TRAIN_DATA_PATH])
    subprocess.run(["dvc", "add", TEST_DATA_PATH])

    # %% 2. Preprocessing
    columns_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'trans_date_trans_time']
    train_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    test_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    class CustomLabelEncoder(LabelEncoder):
        def fit(self, data):
            super().fit(data)
            self.classes_ = np.append(self.classes_, "Unknown")
            return self

        def transform(self, data):
            encoded = np.array([self.classes_.tolist().index(x) if x in self.classes_ else len(self.classes_) - 1 for x in data])
            return encoded

    categorical_features = ['merchant', 'category', 'gender', 'job']
    label_encoders = {}
    label_encoders_dir = os.path.join(MODEL_DIR, 'label_encoders')
    if not os.path.exists(label_encoders_dir):
        os.makedirs(label_encoders_dir)

    for col in categorical_features:
        le = CustomLabelEncoder()
        le.fit(train_data[col])
        train_data[col] = le.transform(train_data[col])
        test_data[col] = le.transform(test_data[col])
        with open(os.path.join(label_encoders_dir, f"{col}_encoder.pkl"), 'wb') as f:
            dill.dump(le, f)

    scaler = StandardScaler()
    train_data['amt'] = scaler.fit_transform(train_data[['amt']])
    test_data['amt'] = scaler.transform(test_data[['amt']])

    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        dill.dump(scaler, f)

    # %% 4. Train the Model
    X_train = train_data.drop(columns=["is_fraud"])
    y_train = train_data["is_fraud"]
    X_test = test_data.drop(columns=["is_fraud"])
    y_test = test_data["is_fraud"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    model_path = os.path.join(MODEL_DIR, f"model_{model_version}.pkl")
    with open(model_path, 'wb') as f:
        dill.dump(model, f)

    subprocess.run(["dvc", "add", model_path])

    # %% 5. Test the Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # %% 6. Save Metrics
    metrics = {
        'model_version': model_version,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'timestamp': datetime.datetime.now().isoformat(),
        'train_data_path': TRAIN_DATA_PATH,
        'test_data_path': TEST_DATA_PATH,
        'hyperparameters': model.get_params()
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(METRICS_DIR, f"metrics_{model_version}.csv")
    metrics_df.to_csv(metrics_file, index=False)
    subprocess.run(["dvc", "add", metrics_file])

    return metrics
