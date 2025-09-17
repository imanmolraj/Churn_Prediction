import axios from "axios";

// ✅ Dynamically set backend API base URL
const API_BASE =
  import.meta.env.VITE_API_BASE || "http://127.0.0.1:5000";

// Single prediction
export function predict(payload) {
  return axios.post(`${API_BASE}/api/predict`, payload);
}

// SHAP explanation for a single prediction
export function explain(payload) {
  return axios.post(`${API_BASE}/api/explain`, payload);
}

// Batch prediction via CSV upload
export function batchPredict(file) {
  const formData = new FormData();
  formData.append("file", file);

  return axios.post(`${API_BASE}/api/batch-predict`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
}

// Dashboard stats (contract, tenure, monthly, metrics)
export function getDashboard() {
  return axios.get(`${API_BASE}/api/dashboard`);
}

// Feature importance for the active model
export function getFeatureImportance() {
  return axios.get(`${API_BASE}/api/feature-importance`);
}
