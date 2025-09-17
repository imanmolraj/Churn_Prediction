import React, { useState } from "react";
import { predict, explain, batchPredict } from "../api";

export default function PredictionForm({ onResult, onExplain, onBatchResult }) {
  const [form, setForm] = useState({
    tenure: 12,
    MonthlyCharges: 50,
    TotalCharges: 600,
    Contract: "Month-to-month",
    InternetService: "DSL",
    PaymentMethod: "Electronic check",
    gender: "Male",
  });
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  function handleChange(e) {
    const { name, value } = e.target;
    setForm((f) => ({ ...f, [name]: value }));
  }

  function handleFileChange(e) {
    setFile(e.target.files[0]);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);

    const payload = {
      ...form,
      tenure: Number(form.tenure),
      MonthlyCharges: Number(form.MonthlyCharges),
      TotalCharges: Number(form.TotalCharges),

      // Default categorical fields required for model
      SeniorCitizen: 0,
      Partner: "No",
      Dependents: "No",
      PhoneService: "Yes",
      MultipleLines: "No",
      OnlineSecurity: "No",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "No",
      StreamingMovies: "No",
      PaperlessBilling: "Yes",
    };

    try {
      // 1️⃣ Predict churn
      const res = await predict(payload);
      onResult(res.data);

      // 2️⃣ Get SHAP explanation
      const exp = await explain(payload);
      onExplain(exp.data);
    } catch (err) {
      console.error("❌ API error:", err.response ? err.response.data : err.message);
      alert("Prediction or explanation failed. Check backend connection.");
    } finally {
      setLoading(false);
    }
  }

  async function handleBatchUpload() {
    if (!file) {
      alert("Please select a CSV file first.");
      return;
    }
    setLoading(true);
    try {
      const res = await batchPredict(file);
      onBatchResult(res.data);
    } catch (err) {
      alert("Batch prediction failed. Check backend connection.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* 🔹 Single Prediction Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <h3 className="text-xl font-semibold mb-4">Enter Customer Details</h3>

        <div>
          <label className="block">Tenure (months)</label>
          <input
            name="tenure"
            type="number"
            value={form.tenure}
            onChange={handleChange}
            className="input"
          />
        </div>

        <div>
          <label className="block">Monthly Charges</label>
          <input
            name="MonthlyCharges"
            type="number"
            value={form.MonthlyCharges}
            onChange={handleChange}
            className="input"
          />
        </div>

        <div>
          <label className="block">Total Charges</label>
          <input
            name="TotalCharges"
            type="number"
            value={form.TotalCharges}
            onChange={handleChange}
            className="input"
          />
        </div>

        <div>
          <label className="block">Contract</label>
          <select
            name="Contract"
            value={form.Contract}
            onChange={handleChange}
            className="input"
          >
            <option>Month-to-month</option>
            <option>One year</option>
            <option>Two year</option>
          </select>
        </div>

        <div>
          <label className="block">Internet Service</label>
          <select
            name="InternetService"
            value={form.InternetService}
            onChange={handleChange}
            className="input"
          >
            <option>DSL</option>
            <option>Fiber optic</option>
            <option>No</option>
          </select>
        </div>

        <div>
          <label className="block">Payment Method</label>
          <select
            name="PaymentMethod"
            value={form.PaymentMethod}
            onChange={handleChange}
            className="input"
          >
            <option>Electronic check</option>
            <option>Mailed check</option>
            <option>Bank transfer (automatic)</option>
            <option>Credit card (automatic)</option>
          </select>
        </div>

        <button type="submit" className="btn w-full" disabled={loading}>
          {loading ? "Processing..." : "Predict & Explain"}
        </button>
      </form>

      {/* 🔹 Batch Upload Section */}
      <div className="border-t pt-4">
        <h3 className="text-xl font-semibold mb-2">Batch Prediction</h3>
        <input type="file" accept=".csv" onChange={handleFileChange} />
        <button
          onClick={handleBatchUpload}
          className="btn w-full mt-2"
          disabled={loading}
        >
          {loading ? "Processing..." : "Upload CSV & Predict"}
        </button>
      </div>
    </div>
  );
}
