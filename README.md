# 🧠 Stroke Prediction using Artificial Neural Network (ANN)

An Artificial Neural Network (ANN) based machine learning project that predicts stroke risk using patient health and lifestyle features such as age, glucose level, BMI, hypertension, heart disease, and smoking status.

### 🚀 Live App

**Try it here 👉 [stroke-prediction-ann-project.streamlit.app](https://stroke-prediction-ann-project.streamlit.app/)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stroke-prediction-ann-project.streamlit.app/)

---

## 📌 Overview

This project focuses on predicting the likelihood of a **stroke occurrence** using a **Machine Learning / Deep Learning approach (Artificial Neural Network)**. The model is trained on medical and lifestyle-related features to classify whether a patient is at risk of stroke.

Early prediction of stroke can help in timely medical intervention and potentially save lives.

---

## 🎯 Problem Statement

Stroke is one of the leading causes of death worldwide. The goal of this project is to build an intelligent system that can:

- Analyze patient health data
- Identify hidden patterns
- Predict stroke risk accurately using an ANN model

---

## 📊 Dataset Features

The dataset includes the following attributes:

- Gender
- Age
- Hypertension
- Heart Disease
- Marital Status
- Work Type
- Residence Type
- Average Glucose Level
- BMI
- Smoking Status
- Stroke (Target Variable)

---

## 🏗️ Model Architecture

This project uses a **Feedforward Artificial Neural Network (ANN)** built with TensorFlow/Keras:

- Input Layer (feature-based input)
- Hidden Layers (ReLU activation)
- Dropout Layers (to reduce overfitting)
- Output Layer (Sigmoid activation for binary classification)

---

## ⚙️ Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Handling Missing Values
4. Encoding Categorical Variables
5. Feature Scaling
6. Handling Class Imbalance (e.g., SMOTE / class weights)
7. Model Building (ANN)
8. Training & Validation
9. Evaluation (Accuracy, Precision, Recall, F1-score)

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

> Special focus is given to **Recall**, because missing a stroke prediction is more critical than false positives.

---

## 🛠️ Technologies Used

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- TensorFlow / Keras
- FastAPI (backend API)
- Streamlit (web app / UI)
- Docker
- Matplotlib & Seaborn

---

## 📁 Project Structure

```
Stroke-prediction-ANN-Projects/
├── Dataset/          # Raw and processed data
├── FastApi/           # FastAPI backend for serving predictions
├── Model/             # Trained ANN model artifacts
├── Project/           # Core project / notebooks / training code
├── Ui/                # Streamlit UI application
├── Dockerfile         # Container build config
├── requirements.txt   # Python dependencies
├── pyproject.toml     # Project metadata
└── README.md
```

---

## 🚀 How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Muhammad-Musharraf/Stroke-prediction-ANN-Projects.git
cd Stroke-prediction-ANN-Projects
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run Ui/app.py
```

### 4️⃣ (Optional) Run the FastAPI Backend

```bash
cd FastApi
uvicorn main:app --reload
```

### 5️⃣ (Optional) Run with Docker

```bash
docker build -t stroke-prediction-app .
docker run -p 8501:8501 stroke-prediction-app
```

> ℹ️ Adjust the file paths/commands above (e.g. `Ui/app.py`, `main:app`) to match the actual entry-point filenames in the `Ui/` and `FastApi/` folders if they differ.

---

## 📊 Key Highlights

- ✔️ End-to-end machine learning pipeline
- ✔️ ANN-based classification model
- ✔️ Handling imbalanced medical dataset
- ✔️ Focus on real-world healthcare prediction problem
- ✔️ Model evaluation with multiple metrics
- ✔️ Deployed as a live Streamlit web app
- ✔️ FastAPI backend for serving predictions

---

## 🧠 Future Improvements

- Hyperparameter tuning for better accuracy
- Integration with real-time patient data
- Model explainability using SHAP / LIME
- CI/CD pipeline for automated deployment

---

## 👤 Author

**Muhammad Musharraf**
Passionate about Artificial Intelligence, Machine Learning, and Deep Learning.

---

## ⭐ Support

If you find this project helpful, please consider giving it a ⭐ on GitHub.
