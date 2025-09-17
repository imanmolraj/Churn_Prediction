import React, { useState } from "react";
import PredictionForm from "./components/PredictionForm";
import Dashboard from "./components/Dashboard";

export default function App() {
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [batchResult, setBatchResult] = useState(null);

  return (
    <div className="container">
      <h1 className="text-3xl font-bold text-center mb-8 text-blue-700">
        Customer Churn Predictor
      </h1>
      <div className="grid md:grid-cols-2 gap-6">
        
        {/* Prediction Form + Results */}
        <div className="card">
          <PredictionForm 
            onResult={setResult} 
            onExplain={setExplanation} 
            onBatchResult={setBatchResult} 
          />

          {/* Single Prediction Result */}
          {result && (
            <div className="mt-6 bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-lg mb-2">Prediction Result</h3>
              <p>
                <strong>Churn Probability:</strong>{" "}
                {(result.probability * 100).toFixed(2)}%
              </p>
              <p>
                <strong>Prediction:</strong>{" "}
                {result.prediction === 1
                  ? "Likely to Churn ❌"
                  : "Not Churning ✅"}
              </p>
            </div>
          )}

          {/* SHAP Explanation */}
          {explanation && (
            <div className="mt-6 bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-lg mb-2">Explanation (SHAP)</h3>
              <pre className="text-sm">{JSON.stringify(explanation, null, 2)}</pre>
            </div>
          )}

          {/* Batch Upload Results */}
          {batchResult && (
            <div className="mt-6 bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-lg mb-2">Batch Prediction Results</h3>
              <pre className="text-sm">{JSON.stringify(batchResult, null, 2)}</pre>
            </div>
          )}
        </div>

        {/* Dashboard */}
        <div className="card">
          <Dashboard />
        </div>
      </div>
    </div>
  );
}
