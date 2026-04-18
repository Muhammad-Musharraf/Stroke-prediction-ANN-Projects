# 🧠 Stroke Prediction using Artificial Neural Network (ANN)

## 📌 Overview

This project focuses on predicting the likelihood of a **stroke occurrence** using a **Machine Learning / Deep Learning approach (Artificial Neural Network)**. The model is trained on medical and lifestyle-related features to classify whether a patient is at risk of stroke.

Early prediction of stroke can help in timely medical intervention and potentially save lives.

---

## 🎯 Problem Statement

Stroke is one of the leading causes of death worldwide. The goal of this project is to build an intelligent system that can:

* Analyze patient health data
* Identify hidden patterns
* Predict stroke risk accurately using an ANN model

---

## 📊 Dataset Features

The dataset includes the following attributes:

* Gender
* Age
* Hypertension
* Heart Disease
* Marital Status
* Work Type
* Residence Type
* Average Glucose Level
* BMI
* Smoking Status
* Stroke (Target Variable)

---

## 🏗️ Model Architecture

This project uses a **Feedforward Artificial Neural Network (ANN)** built with TensorFlow/Keras:

* Input Layer (feature-based input)
* Hidden Layers (ReLU activation)
* Dropout Layers (to reduce overfitting)
* Output Layer (Sigmoid activation for binary classification)

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

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

> Special focus is given to **Recall**, because missing a stroke prediction is more critical than false positives.

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib & Seaborn

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Muhammad-Musharraf/Stroke-prediction-ANN-Projects.git
cd Stroke-prediction-ANN-Projects
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook / Script

```bash
jupyter notebook
```

---

## 📊 Key Highlights

* ✔️ End-to-end machine learning pipeline
* ✔️ ANN-based classification model
* ✔️ Handling imbalanced medical dataset
* ✔️ Focus on real-world healthcare prediction problem
* ✔️ Model evaluation with multiple metrics

---

## 🧠 Future Improvements

* Hyperparameter tuning for better accuracy
* Deployment using Flask / Streamlit
* Integration with real-time patient data
* Model explainability using SHAP / LIME

---

## 👤 Author

**Muhammad Musharraf**
Passionate about Artificial Intelligence, Machine Learning, and Deep Learning.

---

## ⭐ Support

If you find this project helpful, please consider giving it a ⭐ on GitHub.
