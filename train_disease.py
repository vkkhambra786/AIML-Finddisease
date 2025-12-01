# TRAINING DISEASE PREDICTION MODEL
# Works in VS Code + Colab

import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ----------------------------
# 1. Load dataset
# ----------------------------
print("LOG: Loading dataset from breast_cancer.csv...")
df = pd.read_csv('breast_cancer.csv')
X = df.drop('target', axis=1)
y = df['target']
print(f"LOG: Dataset loaded. Shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

# Create output dir early
output_dir = "disease_project_output"
os.makedirs(output_dir, exist_ok=True)

# List patients suffering from cancer in the dataset
print("\n--- LIST OF PATIENTS SUFFERING FROM CANCER IN THE DATASET ---")
malignant_indices = df[df['target'] == 1].index.tolist()
print(f"Total malignant cases: {len(malignant_indices)}")
print("Patient Indices (1-based):", [i+1 for i in malignant_indices])

# Plot malignant distribution
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y)), y, c=y, cmap='coolwarm', alpha=0.7, s=10)
plt.xlabel('Patient Index (0-based)')
plt.ylabel('Target (0=Benign, 1=Malignant)')
plt.title('Distribution of Malignant Cases in the Dataset\nRed = Suffering from Cancer (Malignant), Blue = Not Suffering (Benign)')
plt.grid(True, alpha=0.3)
plot_path = os.path.join(output_dir, 'malignant_distribution.png')
plt.savefig(plot_path)
plt.close()
print(f"LOG: Malignant distribution graph saved as {plot_path}")
output_dir = "disease_project_output"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# 2. Split dataset
# ----------------------------
print("LOG: Splitting dataset into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"LOG: Train set: {X_train.shape}, Test set: {X_test.shape}")

# ----------------------------
# 3. Define Models
# ----------------------------
print("LOG: Defining models...")
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, random_state=42))
])

pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 8, 16],
}
print("LOG: Models defined.")

# ----------------------------
# 4. Train models
# ----------------------------
print("LOG: Training Logistic Regression...")
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
print("LOG: Logistic Regression trained.")

print("LOG: Training Random Forest with GridSearch...")
grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=3, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print(f"LOG: Random Forest trained. Best params: {grid_rf.best_params_}")

# ----------------------------
# 5. Evaluation function
# ----------------------------
def evaluate(model_name, y_true, y_pred, model=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n=== {model_name} ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

    # Plot confusion matrix
    if model is not None:
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['Benign', 'Malignant'], cmap='Blues')
        disp.ax_.set_title(f'Confusion Matrix - {model_name}')
        safe_name = "".join(c for c in model_name if c.isalnum() or c in " _-").replace(" ", "_")
        plot_path = os.path.join(output_dir, f'confusion_matrix_{safe_name}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"LOG: Confusion matrix saved as {plot_path}")

    return f1

f1_lr = evaluate("Logistic Regression", y_test, y_pred_lr, pipe_lr)
f1_rf = evaluate(f"Random Forest {grid_rf.best_params_}", y_test, y_pred_rf, best_rf)

# ----------------------------
# 6. Choose Best Model
# ----------------------------
best_model = best_rf if f1_rf >= f1_lr else pipe_lr
best_name = "RandomForest" if f1_rf >= f1_lr else "LogisticRegression"

print("\nBest model selected:", best_name)

# ----------------------------
# 7. Sample Predictions for 5-8 People
# ----------------------------
print("\nLOG: Generating sample predictions for 6 people...")
sample_data = pd.DataFrame([
    X.mean().to_dict(),  # Average (likely malignant)
    X.min().to_dict(),   # Minimum values (likely benign)
    X.max().to_dict(),   # Maximum values (likely malignant)
    X.quantile(0.25).to_dict(),  # 25th percentile
    X.quantile(0.75).to_dict(),  # 75th percentile
    X.median().to_dict()  # Median
])
sample_names = ['Average Patient', 'Min Features Patient', 'Max Features Patient', '25th Percentile Patient', '75th Percentile Patient', 'Median Patient']

predictions = best_model.predict(sample_data)
probs = best_model.predict_proba(sample_data)[:, 1]  # Probability of malignant

print("Detailed List of Sample Patients:")
print("Patient ID | Name | Prediction | Probability | Why?")
print("-" * 70)
for i, (name, pred, prob) in enumerate(zip(sample_names, predictions, probs)):
    status = "Malignant (Cancerous)" if pred == 1 else "Benign (Not Cancerous)"
    reason = "High tumor features (size, texture)" if pred == 1 else "Low/normal tumor features"
    print(f"{i+1:10} | {name:20} | {status:20} | {prob:.2f} | {reason}")

# Highlight suffering patients
print("\n--- PATIENTS SUFFERING FROM CANCER (Malignant) ---")
malignant_patients = [(i+1, name, prob, reason) for i, (name, pred, prob, reason) in enumerate(zip(sample_names, predictions, probs, [ "High tumor features (size, texture)" if p == 1 else "Low/normal tumor features" for p in predictions])) if pred == 1]
for pid, name, prob, reason in malignant_patients:
    print(f"Patient {pid}: {name} - Probability: {prob:.2f} - Reason: {reason}")

print("\n--- PATIENTS NOT SUFFERING FROM CANCER (Benign) ---")
benign_patients = [(i+1, name, prob, reason) for i, (name, pred, prob, reason) in enumerate(zip(sample_names, predictions, probs, [ "High tumor features (size, texture)" if p == 1 else "Low/normal tumor features" for p in predictions])) if pred == 0]
for pid, name, prob, reason in benign_patients:
    print(f"Patient {pid}: {name} - Probability: {prob:.2f} - Reason: {reason}")

malignant_count = sum(predictions)
benign_count = len(predictions) - malignant_count
print(f"\nTotal Patients: {len(predictions)}")
print(f"Suffering from Cancer (Malignant): {malignant_count}")
print(f"Not Suffering from Cancer (Benign): {benign_count}")

# Plot predictions
plt.figure(figsize=(14, 8))
bars = plt.bar(sample_names, probs, color=['red' if p > 0.5 else 'green' for p in predictions])
plt.axhline(y=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
plt.xlabel('Sample Patients')
plt.ylabel('Probability of Malignant')
plt.title('Disease Prediction for Sample Patients\nRed = Suffering from Cancer (Malignant), Green = Not Suffering (Benign)')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.ylim(0, 1)
for bar, prob, pred in zip(bars, probs, predictions):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{prob:.2f}', ha='center', va='bottom')
    reason = "High tumor features\n(Cancer Risk)" if pred == 1 else "Low/normal features\n(Safe)"
    plt.text(bar.get_x() + bar.get_width()/2, -0.1, reason, ha='center', va='top', fontsize=9, wrap=True)
plt.tight_layout()
plot_path = os.path.join(output_dir, 'sample_predictions.png')
plt.savefig(plot_path)
plt.close()
print(f"LOG: Sample predictions graph saved as {plot_path}")

print("\nLOG: Why predictions? The model uses 30 features (e.g., tumor size, texture). Higher values often indicate malignancy.")

# Summary Graph: Cancer vs No Cancer
malignant_count = sum(predictions)
benign_count = len(predictions) - malignant_count

plt.figure(figsize=(8, 6))
labels = ['Benign (Not Cancerous)', 'Malignant (Cancerous)']
sizes = [benign_count, malignant_count]
colors = ['green', 'red']
explode = (0.1, 0)  # explode benign slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Summary: Cancer Status in Sample Patients')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plot_path = os.path.join(output_dir, 'cancer_summary_pie.png')
plt.savefig(plot_path)
plt.close()
print(f"LOG: Cancer summary pie chart saved as {plot_path}")

print(f"\nSummary: Out of 6 sample patients, {benign_count} are Benign, {malignant_count} are Malignant.")
print("Why? Patients with extreme/high feature values (e.g., large tumors) are predicted malignant.")

# ----------------------------
# 8. Save outputs
# ----------------------------
print("LOG: Saving outputs...")

model_path = os.path.join(output_dir, "disease_model.pkl")
pickle.dump(best_model, open(model_path, "wb"))

# Create sample prediction script
sample_predict_path = os.path.join(output_dir, "sample_predict.py")
with open(sample_predict_path, "w") as f:
    f.write("""
import pickle
import pandas as pd
import os

model_path = os.path.join('disease_project_output', 'disease_model.pkl')
model = pickle.load(open(model_path, 'rb'))
sample = pd.DataFrame([{""" + ', '.join([f"'{c}': {X[c].mean()}" for c in X.columns]) + """}])
pred = model.predict(sample)
print("Prediction:", pred[0])
print("0 = Benign (not cancerous), 1 = Malignant (cancerous)")
""")

# Create README
readme_path = os.path.join(output_dir, "README.txt")
with open(readme_path, "w") as f:
    f.write(f"""
Disease Prediction ML Project
Best Model: {best_name}

Files:
- disease_model.pkl
- sample_predict.py

Run sample_predict.py to see a prediction.
""")

print("\nAll files saved inside:", output_dir)
print("Graphs saved as PNG files in the output directory. Open them to view.")
input("Press Enter to exit...")
