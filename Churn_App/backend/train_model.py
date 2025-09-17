import os
import sys
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, make_scorer
import numpy as np

CSV_PATH = "Telco_Cust_Churn.csv"
MODEL_PATH = "model.pkl"
METRICS_PATH = "metrics.json"

# -------------------
# 1. Check dataset
# -------------------
if not os.path.exists(CSV_PATH):
    print(f"❌ Dataset not found at: {CSV_PATH}")
    sys.exit(1)

print(f"✅ Found dataset: {CSV_PATH}")

# -------------------
# 2. Load & clean
# -------------------
df = pd.read_csv(CSV_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# -------------------
# 3. Train/test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical = X.select_dtypes(include=["object"]).columns.tolist()

# -------------------
# 4. Preprocessor
# -------------------
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# -------------------
# 5. Base models
# -------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
}

# -------------------
# 6. Hyperparameter grids
# -------------------
param_grids = {
    "Logistic Regression": {
        "clf__C": [0.1, 1.0, 10],
        "clf__solver": ["liblinear", "lbfgs"]
    },
    "Random Forest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5]
    },
    "XGBoost": {
        "clf__n_estimators": [200, 300],
        "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0]
    }
}

metrics = {}
best_model = None
best_model_name = None
best_auc = -1

scorer = make_scorer(roc_auc_score, needs_proba=True)

# -------------------
# Helper: Extract feature names
# -------------------
def get_feature_names(preprocessor):
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            encoder = trans
            if isinstance(encoder, OneHotEncoder):
                feature_names.extend(encoder.get_feature_names_out(cols))
    return feature_names

# -------------------
# 7. Train & tune
# -------------------
for name, base_clf in models.items():
    print(f"\n⏳ Tuning {name}...")

    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", base_clf)
    ])

    param_grid = param_grids.get(name, {})
    if param_grid:
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring=scorer, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"🔍 Best params for {name}: {grid.best_params_}")
    else:
        model = pipe.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)

    # Extract feature importance
    feature_importance = {}
    try:
        feature_names = get_feature_names(model.named_steps["pre"])
        clf = model.named_steps["clf"]

        if hasattr(clf, "coef_"):  # Logistic Regression
            importance = np.abs(clf.coef_[0])
        elif hasattr(clf, "feature_importances_"):  # RF, XGB
            importance = clf.feature_importances_
        else:
            importance = None

        if importance is not None:
            feature_importance = {
                fname: round(float(val), 4)
                for fname, val in sorted(
                    zip(feature_names, importance),
                    key=lambda x: -abs(x[1])
                )[:20]  # top 20
            }
    except Exception as e:
        print(f"⚠️ Could not extract feature importance for {name}: {e}")

    metrics[name] = {
        "auc": round(auc, 3),
        "accuracy": round(acc, 3),
        "feature_importance": feature_importance
    }
    print(f"✅ {name}: AUC={auc:.3f}, Acc={acc:.3f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# -------------------
# 8. Save best model & metrics
# -------------------
if best_model is None:
    print("❌ No model trained successfully.")
    sys.exit(1)

joblib.dump(best_model, MODEL_PATH)
print(f"\n🏆 Best model: {best_model_name} (AUC={best_auc:.3f})")
print(f"✅ Saved model → {MODEL_PATH}")

metrics["current_model"] = best_model_name

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"📊 Metrics + feature importance saved → {METRICS_PATH}")
print(json.dumps(metrics, indent=2))
