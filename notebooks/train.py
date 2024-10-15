import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import dill

import datetime
import subprocess
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define paths
TRAIN_DATA_PATH = "../datasets/fraudTrain.csv"
TEST_DATA_PATH = "../datasets/fraudTest.csv"
MODEL_DIR = "../models/"
METRICS_DIR = "../metrics/"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(METRICS_DIR):
    os.makedirs(METRICS_DIR)

# Automatically generate a version string using timestamp
model_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# %% 1. Read and Truncate Data
train_data = pd.read_csv(TRAIN_DATA_PATH).sample(n=30000, random_state=42).reset_index(drop=True)
test_data = pd.read_csv(TEST_DATA_PATH).sample(n=5000, random_state=42).reset_index(drop=True)
print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Track dataset using DVC
subprocess.run(["dvc", "add", TRAIN_DATA_PATH])
subprocess.run(["dvc", "add", TEST_DATA_PATH])

# %% 2. Preprocessing
columns_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'trans_date_trans_time']
train_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Custom LabelEncoder for unseen categories
class CustomLabelEncoder(LabelEncoder):
    def fit(self, data):
        super().fit(data)
        self.classes_ = np.append(self.classes_, "Unknown")
        return self

    def transform(self, data):
        encoded = np.array([self.classes_.tolist().index(x) if x in self.classes_ else len(self.classes_) - 1 for x in data])
        return encoded

# Encode categorical variables
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
    
    # Save encoders using dill
    with open(os.path.join(label_encoders_dir, f"{col}_encoder.pkl"), 'wb') as f:
        dill.dump(le, f)  # Use dill  to save encoders
    print(f"Saved {col} encoder.")



# Similarly for encoders
for col in categorical_features:
    encoder_path = os.path.join(label_encoders_dir, f"{col}_encoder.pkl")
    with open(encoder_path, 'rb') as f:
        loaded_encoder = dill.load(f)
    print(f"{col} encoder loaded successfully after saving.")


# Feature scaling
scaler = StandardScaler()
train_data['amt'] = scaler.fit_transform(train_data[['amt']])
test_data['amt'] = scaler.transform(test_data[['amt']])

# Save the scaler using dill
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    dill.dump(scaler, f)


# Similarly for scaler
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    loaded_scaler = dill.load(f)
print("Scaler loaded successfully after saving.")

# %% 3. Exploratory Data Analysis (Optional)
plt.figure(figsize=(6,6))
plt.pie(train_data["is_fraud"].value_counts(), labels=["No Fraud", "Fraud"], autopct="%0.1f%%", colors=['skyblue', 'salmon'])
plt.title("Fraudulent Transactions Percentage")
# plt.show()

# %% 4. Train the Model
X_train = train_data.drop(columns=["is_fraud"])
y_train = train_data["is_fraud"]
X_test = test_data.drop(columns=["is_fraud"])
y_test = test_data["is_fraud"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Save the trained model using dill
model_path = os.path.join(MODEL_DIR, f"model_{model_version}.pkl")
with open(model_path, 'wb') as f:
    dill.dump(model, f)
print(f"Model saved to {model_path}")

# After saving the model
with open(model_path, 'rb') as f:
    loaded_model = dill.load(f)
print("Model loaded successfully after saving.")



# Track model with DVC
subprocess.run(["dvc", "add", model_path])

# %% 5. Test the Model
y_pred = model.predict(X_test)

# Evaluate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test ROC AUC: {roc_auc:.4f}")

# Save test predictions
test_predictions = test_data.copy()
test_predictions['predicted_fraud'] = y_pred
test_predictions.to_csv(f"../data/test_predictions_{model_version}.csv", index=False)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fraud", "Fraud"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
# plt.show()

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
print("Evaluation metrics saved.")

# Track metrics file with DVC
subprocess.run(["dvc", "add", metrics_file])

# Git tracking and commit
subprocess.run(["git", "add", "../datasets/fraudTrain.csv.dvc", "../datasets/fraudTest.csv.dvc"])
subprocess.run(["git", "add", f"../models/model_{model_version}.pkl.dvc", "../models/.gitignore"])
subprocess.run(["git", "add", f"../metrics/metrics_{model_version}.csv.dvc", "../metrics/.gitignore"])
subprocess.run(["git", "commit", "-m", f"Add new model version {model_version} and associated datasets and metrics"])

# (Optional) Push to the remote repository
# subprocess.run(["git", "push", "origin", "main"])
