# 📊 Customer Churn Prediction Dashboard

An end-to-end **Machine Learning project** for predicting customer churn using **Flask (backend)**, **React + Vite (frontend)**, and **XGBoost/Logistic Regression**.  
The project also includes **EDA + Feature Selection** in Jupyter Notebook for full reproducibility.

---

## 🚀 Features
- Interactive **dashboard** to explore churn behavior:
  - Churn by **Contract type**, **Tenure buckets**, **Monthly charges**
  - Feature importance (from SHAP + model feature importances)
- **Single prediction** → Enter customer details and get churn probability
- **Batch prediction** → Upload CSV and view predictions for multiple customers
- **Model performance comparison** (Accuracy + AUC)
- **Explainability** → SHAP-based model interpretation

---

## 📂 Project Structure

churn-predictor/
├── backend/ 
│ ├── app.py
│ ├── train_model.py
│ ├── model.pkl
│ ├── Telco_Cust_Churn.csv
│ ├── metrics.json
│ └── requirements.txt
├── frontend/ 
│ ├── src/
│ ├── package.json
│ └── vite.config.js
├── notebooks/
│ └── churn_analysis.ipynb
├── README.md
└── .gitignore


---

## ⚙️ Setup Instructions

### 🔹 Backend (Flask)
1. Navigate to backend folder:
   ```bash
   cd backend
2. Create virtual environment & install dependencies:

     python -m venv .venv
     source .venv/bin/activate   
     pip install -r requirements.txt

3.Run the Flask server:

     python app.py

###🔹 Frontend (React + Vite)

1.Navigate to frontend folder:
  cd frontend

2.Install dependencies:
  npm install

3.Start development server:
  npm run dev

### 📒 Notebook

The notebooks/ folder contains:
churn_analysis.ipynb → Includes EDA, feature engineering, model experimentation, and feature selection.
This ensures the full workflow is transparent — from data understanding to deployment.

### 📊 Model Training

To retrain the model:
cd backend
python train_model.py


This generates:
model.pkl → Trained pipeline (preprocessing + model)
metrics.json → Performance metrics + feature importances

## 🛠️ Tech Stack

-> Backend → Python, Flask, Scikit-learn, XGBoost, Pandas, SHAP

-> Frontend → React, Vite, Chart.js, Axios, TailwindCSS

-> Notebook → Jupyter (EDA + Feature Engineering)

