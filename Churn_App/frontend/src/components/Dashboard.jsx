import React, { useEffect, useState } from "react";
import { getDashboard, getFeatureImportance, batchPredict } from "../api";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [batchFI, setBatchFI] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await getDashboard();
        setData(res.data);

        const fi = await getFeatureImportance();
        setFeatureImportance(fi.data);
      } catch (err) {
        console.error(err);
      }
    }
    load();
  }, []);

  async function handleBatchUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const res = await batchPredict(file);
      setBatchResults(res.data.predictions || []);
      setBatchFI(res.data.batch_feature_importance || {});
    } catch (err) {
      alert("Batch prediction failed.");
    }
  }

  if (!data) return <p>Loading dashboard...</p>;

  function makeChart(labels, values, title, color = "#2563eb", horizontal = false) {
    return (
      <div className="card">
        <h4 className="font-semibold mb-2">{title}</h4>
        <Bar
          data={{
            labels,
            datasets: [{ label: title, data: values, backgroundColor: color }],
          }}
          options={{
            indexAxis: horizontal ? "y" : "x",
            responsive: true,
            plugins: { legend: { display: false } },
          }}
        />
      </div>
    );
  }

  // ---- Use metrics from backend ----
  const metrics =
    data.metrics && Object.keys(data.metrics).length > 0
      ? Object.entries(data.metrics)
          .filter(([key]) => key !== "current_model")
          .map(([model, vals]) => ({
            model,
            auc: vals.auc,
            accuracy: vals.accuracy,
          }))
      : [];

  const activeModel = data.active_model || data.metrics?.current_model || null;

  // ---- Order keys properly for charts ----
  const contractOrder = ["Month-to-month", "One year", "Two year"];
  const tenureOrder = ["0-12", "13-24", "25+"];
  const monthlyOrder = ["0-50", "51-100", "101+"];

  const contractLabels = contractOrder.filter((k) => k in data.contract);
  const contractValues = contractLabels.map((k) =>
    (data.contract[k] * 100).toFixed(2)
  );

  const tenureLabels = tenureOrder.filter((k) => k in data.tenure);
  const tenureValues = tenureLabels.map((k) =>
    (data.tenure[k] * 100).toFixed(2)
  );

  const monthlyLabels = monthlyOrder.filter((k) => k in data.monthly);
  const monthlyValues = monthlyLabels.map((k) =>
    (data.monthly[k] * 100).toFixed(2)
  );

  // ---- Feature importance (global model) ----
  const fiLabels = featureImportance
    ? Object.keys(featureImportance.feature_importance || {})
    : [];
  const fiValues = featureImportance
    ? Object.values(featureImportance.feature_importance || {})
    : [];

  // ---- Batch Feature Importance ----
  const batchFILabels = batchFI ? Object.keys(batchFI) : [];
  const batchFIValues = batchFI ? Object.values(batchFI) : [];

  return (
    <div className="space-y-6">
      {/* Active Model Banner */}
      {activeModel && (
        <div className="p-4 bg-blue-100 border border-blue-300 rounded-lg shadow-sm">
          <p className="text-blue-800 font-semibold">
            ✅ Currently deployed model:{" "}
            <span className="font-bold">{activeModel}</span>
          </p>
        </div>
      )}

      {/* Model comparison table */}
      <div className="card p-4">
        <h3 className="text-lg font-semibold mb-3">
          Model Performance Comparison
        </h3>
        <table className="w-full border-collapse">
          <thead>
            <tr className="text-left border-b">
              <th className="p-2">Model</th>
              <th className="p-2">AUC</th>
              <th className="p-2">Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m, i) => {
              const isActive = activeModel && m.model === activeModel;
              return (
                <tr
                  key={i}
                  className={`border-b ${isActive ? "bg-blue-50 font-bold" : ""}`}
                >
                  <td className="p-2">
                    {m.model}
                    {isActive && " (Active)"}
                  </td>
                  <td className="p-2">{m.auc}</td>
                  <td className="p-2">{m.accuracy}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Charts from backend data */}
      {makeChart(contractLabels, contractValues, "Churn by Contract")}
      {makeChart(tenureLabels, tenureValues, "Churn by Tenure")}
      {makeChart(monthlyLabels, monthlyValues, "Churn by Monthly Charges")}

      {/* Feature Importance */}
      {fiLabels.length > 0 &&
        makeChart(fiLabels, fiValues, "Global Feature Importance", "#16a34a", true)}

      {/* Batch Prediction */}
      <div className="card p-4">
        <h3 className="text-lg font-semibold mb-3">
          Batch Prediction (Upload CSV)
        </h3>
        <input type="file" accept=".csv" onChange={handleBatchUpload} />

        {/* Batch Results Table */}
        {batchResults && batchResults.length > 0 && (
          <table className="w-full border mt-4 text-sm">
            <thead>
              <tr className="bg-gray-100 border-b">
                <th className="p-2">Customer</th>
                <th className="p-2">Prediction</th>
                <th className="p-2">Probability</th>
              </tr>
            </thead>
            <tbody>
              {batchResults.map((row, i) => (
                <tr key={i} className="border-b">
                  <td className="p-2">{row.CustomerID || `Row ${i + 1}`}</td>
                  <td className="p-2">
                    {row.prediction === 1 ? "Churn ❌" : "No Churn ✅"}
                  </td>
                  <td className="p-2">{(row.probability * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* Batch Feature Importance */}
        {batchFILabels.length > 0 &&
          makeChart(
            batchFILabels,
            batchFIValues,
            "Batch Feature Importance (SHAP)",
            "#f59e0b",
            true
          )}
      </div>
    </div>
  );
}
