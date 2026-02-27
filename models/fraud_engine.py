"""
models/fraud_engine.py

Three ML components:
  1. FraudClassifier     — Random Forest binary classifier (is_fraud: 0/1)
  2. AnomalyDetector     — Isolation Forest per-employee behavioural anomaly
  3. RiskScorer          — Combines both outputs into a 0–100 risk score

Usage:
  engine = FraudEngine()
  engine.train("data/expenses_training.csv")
  result = engine.predict(expense_dict)
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

# ── Constants ──────────────────────────────────────────────────────────────────
POLICY_LIMITS = {
    "Travel": 15000, "Food": 2000, "Hotel": 8000,
    "Fuel": 3000, "Office Supplies": 5000, "Entertainment": 4000,
}

MODEL_PATH = "models/saved/"

# Features used by the Random Forest classifier
CLASSIFIER_FEATURES = [
    "amount_over_limit",
    "exceed_ratio",
    "is_weekend",
    "is_duplicate",
    "recent_submissions",
    "amount_vs_avg",
    "has_receipt",
    "day_of_week",
    "category_enc",
    "payment_mode_enc",
]

# Features used by Isolation Forest (behavioural/spending patterns)
ANOMALY_FEATURES = [
    "amount",
    "exceed_ratio",
    "recent_submissions",
    "amount_vs_avg",
    "is_weekend",
    "has_receipt",
]


# ── Encoders (fitted at train time, reused at predict time) ───────────────────
class FraudEngine:
    def __init__(self):
        self.classifier     = None   # RandomForestClassifier
        self.anomaly_model  = None   # IsolationForest (global)
        self.emp_models     = {}     # {emp_id: IsolationForest}
        self.cat_encoder    = LabelEncoder()
        self.pay_encoder    = LabelEncoder()
        self.scaler         = StandardScaler()
        self.is_trained     = False

    # ── Training ───────────────────────────────────────────────────────────────
    def train(self, csv_path: str):
        df = pd.read_csv(csv_path)
        print(f"\n[FraudEngine] Training on {len(df)} records …")

        df = self._engineer_features(df, fit=True)

        # ── 1. Random Forest Classifier ──────────────────────────────────────
        X = df[CLASSIFIER_FEATURES].values
        y = df["is_fraud"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",   # handles class imbalance
            random_state=42,
        )
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test)[:, 1]

        print("\n[Classifier] Test set performance:")
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
        print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

        # Feature importance
        importances = zip(CLASSIFIER_FEATURES, self.classifier.feature_importances_)
        print("\n[Classifier] Feature importances:")
        for feat, imp in sorted(importances, key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"  {feat:<25} {bar} {imp:.3f}")

        # ── 2. Global Isolation Forest (overall anomaly detection) ────────────
        X_anom = df[ANOMALY_FEATURES].values
        X_anom = self.scaler.fit_transform(X_anom)

        self.anomaly_model = IsolationForest(
            n_estimators=200,
            contamination=0.1,   # expect ~10% anomalous
            random_state=42,
        )
        self.anomaly_model.fit(X_anom)
        print("\n[IsolationForest] Global anomaly model trained.")

        # ── 3. Per-employee Isolation Forest ─────────────────────────────────
        for emp_id, grp in df.groupby("employee_id"):
            if len(grp) >= 10:   # need enough samples
                X_emp = grp[ANOMALY_FEATURES].values
                X_emp = StandardScaler().fit_transform(X_emp)
                iso = IsolationForest(
                    n_estimators=100,
                    contamination=0.15,
                    random_state=42,
                )
                iso.fit(X_emp)
                self.emp_models[emp_id] = iso
        print(f"[IsolationForest] Per-employee models: {list(self.emp_models.keys())}")

        self.is_trained = True
        self._save()
        print("\n[FraudEngine] Training complete. Models saved.")

    # ── Prediction ─────────────────────────────────────────────────────────────
    def predict(self, expense: dict) -> dict:
        """
        expense dict keys:
          employee_id, category, amount, vendor, date (YYYY-MM-DD),
          payment_mode, has_receipt (bool), recent_submissions (int),
          avg_monthly (float)
        Returns:
          fraud_probability (0–1), anomaly_score (0–100), risk_score (0–100),
          status, flags, feature_values
        """
        if not self.is_trained:
            self._load()

        row = self._build_row(expense)
        df  = pd.DataFrame([row])
        df  = self._engineer_features(df, fit=False)

        # ── Fraud probability (Random Forest) ────────────────────────────────
        X_clf = df[CLASSIFIER_FEATURES].values
        fraud_prob = float(self.classifier.predict_proba(X_clf)[0, 1])

        # ── Global anomaly score (Isolation Forest) ───────────────────────────
        X_anom = self.scaler.transform(df[ANOMALY_FEATURES].values)
        # decision_function: more negative = more anomalous
        raw_global = float(self.anomaly_model.decision_function(X_anom)[0])
        anomaly_global = self._normalize_anomaly(raw_global)

        # ── Per-employee anomaly score ────────────────────────────────────────
        emp_id = expense.get("employee_id", "")
        if emp_id in self.emp_models:
            X_emp = StandardScaler().fit_transform(df[ANOMALY_FEATURES].values)
            raw_emp = float(self.emp_models[emp_id].decision_function(X_emp)[0])
            anomaly_emp = self._normalize_anomaly(raw_emp)
        else:
            anomaly_emp = anomaly_global   # fallback

        # Combine: weighted average (employee-level is more informative)
        anomaly_score = round(0.4 * anomaly_global + 0.6 * anomaly_emp, 1)

        # ── Rule-based flags ──────────────────────────────────────────────────
        flags = self._compute_flags(expense, row)

        # ── Final risk score (0–100) ──────────────────────────────────────────
        # Weighted blend: fraud prob (strongest signal) + anomaly + flags
        flag_penalty = min(len(flags) * 12, 36)
        risk_score   = round(
            fraud_prob * 55          # ML fraud signal    (0–55)
            + anomaly_score * 0.30   # behavioural anomaly (0–30)
            + flag_penalty           # hard rule violations (0–36, capped)
        )
        risk_score = int(min(risk_score, 100))

        # ── Status classification ─────────────────────────────────────────────
        if risk_score < 30:
            status = "Approved"
        elif risk_score < 65:
            status = "Needs Review"
        else:
            status = "High Risk"

        return {
            "fraud_probability": round(fraud_prob, 4),
            "anomaly_score":     round(anomaly_score, 1),
            "risk_score":        risk_score,
            "status":            status,
            "flags":             flags,
            "feature_values": {
                "exceed_ratio":       round(row["exceed_ratio"], 3),
                "amount_vs_avg":      round(row["amount_vs_avg"], 3),
                "recent_submissions": row["recent_submissions"],
                "is_weekend":         bool(row["is_weekend"]),
                "is_duplicate":       bool(row.get("is_duplicate", 0)),
                "has_receipt":        bool(row["has_receipt"]),
            },
        }

    def predict_employee_profile(self, emp_id: str, expenses: list) -> dict:
        """
        Compute aggregated risk profile for one employee from a list of
        their expense dicts (already scored via predict()).
        """
        if not expenses:
            return {"anomaly_score": 0, "is_anomaly": False, "summary": {}}

        scores      = [e["risk_score"]        for e in expenses]
        fraud_probs = [e["fraud_probability"]  for e in expenses]
        flags_all   = [f for e in expenses for f in e.get("flags", [])]

        avg_risk    = round(np.mean(scores), 1)
        max_risk    = max(scores)
        avg_fraud_p = round(np.mean(fraud_probs), 4)
        flag_rate   = round(
            sum(1 for e in expenses if e.get("flags")) / len(expenses), 3
        )

        # Behavioural anomaly: combine avg fraud prob + flag rate
        anomaly_score = round(
            min(100, avg_fraud_p * 60 + flag_rate * 40 + (max_risk * 0.1))
        )
        is_anomaly = anomaly_score > 55

        from collections import Counter
        top_flags = Counter(flags_all).most_common(3)

        return {
            "employee_id":    emp_id,
            "anomaly_score":  anomaly_score,
            "is_anomaly":     is_anomaly,
            "avg_risk_score": avg_risk,
            "max_risk_score": max_risk,
            "avg_fraud_prob": avg_fraud_p,
            "flag_rate":      flag_rate,
            "total_expenses": len(expenses),
            "top_flags":      [{"flag": f, "count": c} for f, c in top_flags],
        }

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _build_row(self, expense: dict) -> dict:
        amount    = float(expense["amount"])
        category  = expense.get("category", "Travel")
        limit     = POLICY_LIMITS.get(category, 5000)
        avg_monthly = float(expense.get("avg_monthly", 5000))

        date_str  = expense.get("date", "2025-01-01")
        date_obj  = pd.to_datetime(date_str)
        is_weekend = int(date_obj.weekday() >= 5)

        return {
            "employee_id":      expense.get("employee_id", ""),
            "category":         category,
            "amount":           amount,
            "policy_limit":     limit,
            "vendor":           expense.get("vendor", ""),
            "day_of_week":      date_obj.weekday(),
            "payment_mode":     expense.get("payment_mode", "Credit Card"),
            "has_receipt":      int(bool(expense.get("has_receipt", True))),
            "avg_monthly":      avg_monthly,
            "is_weekend":       is_weekend,
            "is_duplicate":     int(bool(expense.get("is_duplicate", False))),
            "recent_submissions": int(expense.get("recent_submissions", 0)),
            # derived
            "amount_over_limit": int(amount > limit),
            "exceed_ratio":      round(amount / limit, 4),
            "amount_vs_avg":     round(amount / avg_monthly, 4),
        }

    def _engineer_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()

        if fit:
            df["category_enc"]     = self.cat_encoder.fit_transform(df["category"])
            df["payment_mode_enc"] = self.pay_encoder.fit_transform(df["payment_mode"])
        else:
            # Handle unseen labels gracefully
            def safe_transform(enc, values):
                known = set(enc.classes_)
                return np.array([
                    enc.transform([v])[0] if v in known else -1
                    for v in values
                ])
            df["category_enc"]     = safe_transform(self.cat_encoder, df["category"])
            df["payment_mode_enc"] = safe_transform(self.pay_encoder, df["payment_mode"])

        return df

    def _compute_flags(self, expense: dict, row: dict) -> list:
        flags = []
        if row["amount_over_limit"]:
            flags.append("Exceeds Policy Limit")
        if row["is_weekend"]:
            flags.append("Weekend Claim")
        if not row["has_receipt"]:
            flags.append("Receipt Missing")
        if row.get("is_duplicate"):
            flags.append("Duplicate Claim")
        if row["recent_submissions"] > 8:
            flags.append("Unusually Frequent Submissions")
        if row["exceed_ratio"] > 1.8:
            flags.append("Severely Over Limit")
        if row["amount_vs_avg"] > 0.5:
            flags.append("High Spend vs Monthly Average")
        return flags

    @staticmethod
    def _normalize_anomaly(raw_score: float) -> float:
        """
        Isolation Forest decision_function typically ranges ~ -0.5 to +0.5.
        We flip and scale to 0–100 where 100 = most anomalous.
        """
        clipped = max(-0.5, min(0.5, raw_score))
        return round((0.5 - clipped) * 100, 1)

    # ── Persistence ────────────────────────────────────────────────────────────
    def _save(self):
        os.makedirs(MODEL_PATH, exist_ok=True)
        with open(f"{MODEL_PATH}engine.pkl", "wb") as f:
            pickle.dump(self, f)
        print(f"[FraudEngine] Saved to {MODEL_PATH}engine.pkl")

    def _load(self):
        path = f"{MODEL_PATH}engine.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained model found at {path}. "
                "Run train.py first."
            )
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)
        self.is_trained = True
        print("[FraudEngine] Model loaded from disk.")
