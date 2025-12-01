# AIML-FindDisease: Breast Cancer Prediction Model

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning project for predicting breast cancer (malignant vs. benign) using the Wisconsin Breast Cancer Dataset. This project demonstrates data preprocessing, model training, evaluation, and visualization to identify patients at risk.

## 📋 Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features
- **Data Analysis**: Loads and analyzes breast cancer dataset with 30 features
- **Model Training**: Trains Logistic Regression and Random Forest models with hyperparameter tuning
- **Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1-score) and confusion matrices
- **Predictions**: Generates sample predictions with probabilities and explanations
- **Visualization**: Multiple graphs including confusion matrices, prediction charts, and cancer distribution
- **Patient Lists**: Identifies and lists patients suffering from cancer from the dataset
- **Exportable Model**: Saves trained model for future predictions

## 🔍 How It Works

### Measurements for Cancer Detection
The model uses **30 numerical features** derived from digitized breast tumor images:

#### Feature Categories
- **Mean Values**: Average measurements across the tumor
- **Standard Error**: Variation in measurements
- **Worst Values**: Maximum measurements

#### Key Measurements
- **Radius**: Distance from center to perimeter points
- **Texture**: Gray-scale value variations
- **Perimeter**: Tumor boundary length
- **Area**: Tumor size in pixels
- **Smoothness**: Edge variations
- **Compactness**: Density (perimeter²/area)
- **Concavity**: Inward curve severity
- **Concave Points**: Number of indentations
- **Symmetry**: Shape symmetry
- **Fractal Dimension**: Complexity measure

### Prediction Process
1. **Data Loading**: Reads CSV with 569 patient records
2. **Preprocessing**: Standardizes features using StandardScaler
3. **Model Training**: Trains on 80% of data, validates on 20%
4. **Prediction**: For new data, outputs probability of malignancy
5. **Threshold**: Probability > 0.5 = Malignant (cancer), ≤ 0.5 = Benign

### Why It Detects Cancer
- **Higher values** in size features (radius, area) often indicate malignancy
- **Texture and concavity** patterns correlate with cancerous tumors
- Model learns from labeled medical data to identify risk patterns

## 📊 Dataset
- **Source**: Wisconsin Breast Cancer Dataset (scikit-learn)
- **Size**: 569 patients, 30 features each
- **Target**: Binary classification (0 = Benign, 1 = Malignant)
- **Distribution**: 357 malignant (62.7%), 212 benign (37.3%)

## 🤖 Model Details

### Algorithms Used
1. **Logistic Regression** (Selected as best model)
   - Accuracy: 98.2%
   - Uses sigmoid function for probability estimation

2. **Random Forest** (Alternative)
   - Accuracy: 95.6%
   - Ensemble of decision trees with hyperparameter tuning

### Training Process
- **Data Split**: 80% training, 20% testing (stratified)
- **Scaling**: StandardScaler for feature normalization
- **Grid Search**: Optimizes Random Forest parameters
- **Evaluation**: F1-score used for model selection

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/vkkhambra786/AIML-Finddisease.git
cd AIML-Finddisease

# Create virtual environment
python -m venv .venv
# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirement.txt
```

## 📖 Usage

### Run the Main Script
```bash
python train_disease.py
```

This will:
- Load and analyze the dataset
- Train models and display metrics
- Generate predictions and visualizations
- Save outputs to `disease_project_output/`

### Make New Predictions
Use the generated `sample_predict.py`:
```python
import pickle
import pandas as pd

# Load model
model = pickle.load(open('disease_project_output/disease_model.pkl', 'rb'))

# Prepare new data (30 features required)
new_data = pd.DataFrame([{
    'mean radius': 14.0,
    'mean texture': 20.0,
    # ... add all 30 features
}])

# Predict
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[0][1]

print(f"Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")
print(f"Probability of Cancer: {probability:.2f}")
```

## 📈 Outputs

The script generates comprehensive outputs in `disease_project_output/`:

### Logs
- Dataset statistics and distribution
- Model training progress
- Evaluation metrics for both models
- List of patients suffering from cancer (357 indices)
- Sample predictions with explanations

### Files Generated
- `disease_model.pkl`: Trained model for predictions
- `sample_predict.py`: Script for new predictions
- `README.txt`: Basic usage instructions

### Visualizations (PNG files)
1. **Confusion Matrices**: For Logistic Regression and Random Forest
2. **Sample Predictions Bar Chart**: Shows probabilities for 6 synthetic patients
3. **Cancer Summary Pie Chart**: Malignant vs benign distribution
4. **Malignant Distribution Scatter Plot**: Visualizes all 569 patients (red = cancer, blue = benign)

## 📊 Results

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 98.2% | 98.6% | 98.6% | 98.6% |
| Random Forest | 95.6% | 95.9% | 97.2% | 96.5% |

### Key Insights
- Logistic Regression selected as best model
- 357 patients identified as suffering from cancer
- Model correctly identifies risk based on tumor measurements
- Higher feature values correlate with malignancy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

 

**Disclaimer**: This is an educational ML project using public dataset features. Not intended for medical diagnosis. Consult healthcare professionals for actual medical advice.
