"""
train.py
Run this once to generate data and train all ML models.

    python train.py
"""
import subprocess
import sys
import os

# ── Step 1: generate training data ───────────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating synthetic training data …")
print("=" * 60)
exec(open("data/generate_training_data.py").read())

# ── Step 2: train the fraud engine ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Training ML models …")
print("=" * 60)

from models.fraud_engine import FraudEngine

engine = FraudEngine()
engine.train("data/expenses_training.csv")

print("\n" + "=" * 60)
print("STEP 3: Smoke-test prediction …")
print("=" * 60)

# Test with a clearly fraudulent expense
fraud_expense = {
    "employee_id":       "E002",
    "category":          "Travel",
    "amount":            24000,          # way over 15000 limit
    "vendor":            "MakeMyTrip",
    "date":              "2025-06-07",   # Saturday
    "payment_mode":      "Cash",
    "has_receipt":       False,
    "recent_submissions": 12,
    "avg_monthly":       7800,
    "is_duplicate":      True,
}

# Test with a clean legitimate expense
clean_expense = {
    "employee_id":       "E003",
    "category":          "Food",
    "amount":            850,
    "vendor":            "Swiggy",
    "date":              "2025-06-10",   # Tuesday
    "payment_mode":      "UPI",
    "has_receipt":       True,
    "recent_submissions": 1,
    "avg_monthly":       2100,
    "is_duplicate":      False,
}

print("\n── Fraudulent Expense ──")
r1 = engine.predict(fraud_expense)
print(f"  Risk Score:        {r1['risk_score']}")
print(f"  Status:            {r1['status']}")
print(f"  Fraud Probability: {r1['fraud_probability']:.1%}")
print(f"  Anomaly Score:     {r1['anomaly_score']}")
print(f"  Flags:             {r1['flags']}")

print("\n── Clean Expense ──")
r2 = engine.predict(clean_expense)
print(f"  Risk Score:        {r2['risk_score']}")
print(f"  Status:            {r2['status']}")
print(f"  Fraud Probability: {r2['fraud_probability']:.1%}")
print(f"  Anomaly Score:     {r2['anomaly_score']}")
print(f"  Flags:             {r2['flags']}")

print("\n✅  Training and smoke test complete.")
print("    Run 'python app.py' to start the API server.")
