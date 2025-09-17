from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import json
import shap

app = Flask(__name__)
CORS(app)

# ----------------------
# Paths
# ----------------------
MODEL_PATH = "model.pkl"
DATA_PATH = "Telco_Cust_Churn.csv"
METRICS_PATH = "metrics.json"

# ----------------------
# Load model
# ----------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ {MODEL_PATH} not found. Please run train_model.py first."
    )

model = joblib.load(MODEL_PATH)  # this is a pipeline (pre + clf)
print("✅ Model loaded successfully.")

# ----------------------
# Load dataset for dashboard
# ----------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"❌ {DATA_PATH} not found. Please place Telco_Cust_Churn.csv in backend/."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ----------------------
# Routes
# ----------------------
@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check"""
    active_model = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
            active_model = metrics.get("current_model")

    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "active_model": active_model
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Run churn prediction for a single customer"""
    try:
        data = request.json
        df_input = pd.DataFrame([data])

        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]

        return jsonify({
            "prediction": int(pred),
            "probability": float(prob)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    """Return churn distributions + model metrics"""

    # ---- Contract ----
    contract_rates = df.groupby("Contract")["Churn"].mean().to_dict()

    # ---- Tenure buckets ----
    def tenure_bucket(t):
        if t <= 12:
            return "0-12"
        elif t <= 24:
            return "13-24"
        else:
            return "25+"

    df["TenureBucket"] = df["tenure"].apply(tenure_bucket)
    tenure_rates = df.groupby("TenureBucket")["Churn"].mean().to_dict()

    # ---- Monthly charges buckets ----
    def monthly_bucket(m):
        if m <= 50:
            return "0-50"
        elif m <= 100:
            return "51-100"
        else:
            return "101+"

    df["MonthlyBucket"] = df["MonthlyCharges"].apply(monthly_bucket)
    monthly_rates = df.groupby("MonthlyBucket")["Churn"].mean().to_dict()

    # ---- Load model metrics ----
    metrics = {}
    active_model = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
            active_model = metrics.get("current_model")

    return jsonify({
        "contract": contract_rates,
        "tenure": tenure_rates,
        "monthly": monthly_rates,
        "metrics": metrics,
        "active_model": active_model
    })


@app.route("/api/feature-importance", methods=["GET"])
def feature_importance():
    """Return feature importance for the active model"""
    feature_importance = {}
    active_model = None

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
            active_model = metrics.get("current_model")

            if active_model and active_model in metrics:
                feature_importance = metrics[active_model].get("feature_importance", {})

    return jsonify({
        "active_model": active_model,
        "feature_importance": feature_importance
    })


@app.route("/api/explain", methods=["POST"])
def explain():
    """Explain prediction for a single customer using SHAP"""
    try:
        data = request.json
        df_input = pd.DataFrame([data])

        # Apply preprocessing before SHAP
        X_transformed = model.named_steps["pre"].transform(df_input)
        clf = model.named_steps["clf"]

        # Use SHAP explainer for tree/linear models
        explainer = shap.Explainer(clf, X_transformed)
        shap_values = explainer(X_transformed)

        explanation = {
            "features": df_input.to_dict(orient="records")[0],
            "shap_values": shap_values.values[0].tolist(),
            "base_value": float(shap_values.base_values[0])
        }

        return jsonify(explanation)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """Run predictions for multiple customers via CSV upload and summarize with SHAP"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        print("📂 Received file:", file.filename)

        # ---- Load input CSV ----
        df_input = pd.read_csv(file)
        df_input.columns = df_input.columns.str.strip()

        if df_input.empty:
            return jsonify({"error": "Uploaded CSV is empty"}), 400

        # ---- Predictions ----
        preds = model.predict(df_input).tolist()
        probs = model.predict_proba(df_input)[:, 1].tolist()

        results = df_input.copy()
        results["prediction"] = preds
        results["probability"] = probs

        # ---- SHAP batch explanation ----
        feature_importance = {}
        try:
            X_transformed = model.named_steps["pre"].transform(df_input)
            clf = model.named_steps["clf"]

            # Initialize SHAP explainer (safe for tree/linear models)
            explainer = shap.Explainer(clf, X_transformed)
            shap_values = explainer(X_transformed)

            # Mean absolute SHAP values across all rows in batch
            mean_abs_shap = shap_values.abs.mean(0).tolist()
            feature_names = model.named_steps["pre"].get_feature_names_out()

            feature_importance = dict(zip(feature_names, mean_abs_shap))
        except Exception as shap_err:
            print("⚠️ SHAP explanation failed:", shap_err)

        # ---- Response ----
        return jsonify({
            "predictions": results.to_dict(orient="records"),  # ✅ proper JSON
            "batch_feature_importance": feature_importance
        })

    except Exception as e:
        print("❌ Batch prediction error:", str(e))
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=5000, debug=True)
