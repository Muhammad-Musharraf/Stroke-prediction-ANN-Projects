# 🧠 Stroke Prediction — Artificial Neural Network

> An end-to-end deep learning system that predicts stroke risk from patient health and lifestyle data, served through a production-ready FastAPI backend and containerized with Docker.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Dataset & Features](#dataset--features)
- [Model Architecture](#model-architecture)
- [ML Pipeline](#ml-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [API Reference](#api-reference)
- [Future Work](#future-work)
- [Author](#author)

---

## Overview

Stroke is among the leading causes of death and long-term disability worldwide. This project builds an intelligent, data-driven system that:

- Ingests structured patient health records
- Processes and normalises the data through a reproducible ML pipeline
- Classifies stroke risk using a trained Artificial Neural Network (ANN)
- Exposes predictions through a REST API (FastAPI) with an accompanying UI

The goal is to support early clinical decision-making by surfacing high-risk patients before a stroke event occurs.

---

## Problem Statement

Given a set of demographic and clinical features — such as age, BMI, glucose levels, and smoking history — predict whether a patient is at risk of suffering a stroke (binary classification: `0 = No Stroke`, `1 = Stroke`).

Because medical false negatives carry higher costs than false positives, the model is evaluated with particular emphasis on **Recall**.

---

## Project Structure

```
Stroke-prediction-ANN-Projects/
│
├── Dataset/            # Raw and processed data files
├── FastApi/            # FastAPI application (routes, schemas, inference logic)
├── Model/              # Saved trained model and preprocessing artifacts
├── Project/            # Notebooks, EDA, and training scripts
├── Ui/                 # Frontend interface for submitting predictions
│
├── Dockerfile          # Container definition (Python 3.12-slim, port 8000)
├── requirements.txt    # Pinned Python dependencies
├── pyproject.toml      # Project metadata
├── uv.lock             # Lockfile (uv package manager)
└── .env                # Environment variables (not committed — see setup)
```

---

## Dataset & Features

The dataset is based on real-world medical records containing the following attributes:

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Male / Female / Other |
| `age` | Numerical | Patient age in years |
| `hypertension` | Binary | 0 = No, 1 = Yes |
| `heart_disease` | Binary | 0 = No, 1 = Yes |
| `ever_married` | Categorical | Yes / No |
| `work_type` | Categorical | Private, Self-employed, Govt, Children, Never worked |
| `Residence_type` | Categorical | Urban / Rural |
| `avg_glucose_level` | Numerical | Average blood glucose (mg/dL) |
| `bmi` | Numerical | Body Mass Index |
| `smoking_status` | Categorical | Never, Formerly smoked, Smokes, Unknown |
| `stroke` | Binary | **Target** — 0 = No Stroke, 1 = Stroke |

> **Class Imbalance Note:** The dataset is heavily imbalanced (stroke cases are a minority). This is handled via class weighting or oversampling (SMOTE) during training.

---

## Model Architecture

A Feedforward Artificial Neural Network built with **TensorFlow / Keras**:

```
Input Layer  →  [N features after encoding & scaling]
     ↓
Dense Layer  →  ReLU activation
     ↓
Dropout Layer  →  Regularisation (reduce overfitting)
     ↓
Dense Layer  →  ReLU activation
     ↓
Dropout Layer
     ↓
Output Layer →  1 neuron, Sigmoid activation (binary probability)
```

- **Loss function:** Binary Crossentropy
- **Optimizer:** Adam
- **Output:** Probability score ∈ [0, 1]; thresholded at 0.5 for classification

---

## ML Pipeline

```
1. Data Loading & Exploration (EDA)
        ↓
2. Handling Missing Values  (e.g., BMI imputation)
        ↓
3. Encoding Categorical Variables  (Label / One-Hot encoding)
        ↓
4. Feature Scaling  (StandardScaler / MinMaxScaler)
        ↓
5. Addressing Class Imbalance  (SMOTE / class_weight)
        ↓
6. Train / Validation / Test Split
        ↓
7. ANN Model Training
        ↓
8. Evaluation & Threshold Tuning
        ↓
9. Model & Scaler Serialization  (Keras + joblib)
        ↓
10. FastAPI Deployment
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted strokes, how many were real |
| **Recall** | Of actual strokes, how many were caught *(primary metric)* |
| **F1 Score** | Harmonic mean of Precision & Recall |
| **Confusion Matrix** | Breakdown of TP, TN, FP, FN |

> Recall is prioritised — a missed stroke prediction is far more dangerous than a false alarm.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Deep Learning | TensorFlow 2.20 / Keras 3.13 |
| ML Utilities | Scikit-learn 1.6, SciPy 1.14 |
| Data Processing | Pandas 2.2, NumPy |
| API Server | FastAPI 0.115, Uvicorn 0.34 |
| Serialization | Joblib 1.4 |
| Monitoring | Sentry SDK 2.13 |
| Containerization | Docker (Python 3.12-slim) |
| Dependency Manager | uv |

---

## Getting Started

### Prerequisites

- Python 3.12+
- `pip` or [`uv`](https://github.com/astral-sh/uv)
- Docker (for containerized deployment)

---

### Local Setup

**1. Clone the repository**

```bash
git clone https://github.com/Muhammad-Musharraf/Stroke-prediction-ANN-Projects.git
cd Stroke-prediction-ANN-Projects
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

*Or with uv:*

```bash
uv sync
```

**4. Configure environment variables**

Copy the example env file and fill in any required values:

```bash
cp .env .env.local
```

**5. Run the FastAPI server**

```bash
uvicorn FastApi.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

### Docker Setup

**Build the image**

```bash
docker build -t stroke-prediction-api .
```

**Run the container**

```bash
docker run -p 8000:8000 --env-file .env stroke-prediction-api
```

The API will be available at `http://localhost:8000`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Submit patient features, receive stroke risk prediction |
| `GET` | `/docs` | Swagger UI — interactive API documentation |

**Example request body (`POST /predict`):**

```json
{
  "gender": "Male",
  "age": 67,
  "hypertension": 0,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

**Example response:**

```json
{
  "stroke_probability": 0.82,
  "prediction": 1,
  "risk_level": "High"
}
```

---

## Future Work

- [ ] Hyperparameter tuning (learning rate, dropout rate, layer depth) via Keras Tuner
- [ ] Cross-validation for more robust evaluation
- [ ] Model explainability with SHAP or LIME
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Deployment to cloud (AWS ECS / GCP Cloud Run / Railway)
- [ ] Expanded monitoring and alerting via Sentry

---

## Author

**Muhammad Musharraf**  
Passionate about Artificial Intelligence, Machine Learning, and Deep Learning.

- GitHub: [@Muhammad-Musharraf](https://github.com/Muhammad-Musharraf)

---

*If you find this project useful, consider giving it a ⭐ on GitHub — it helps others discover it.*
