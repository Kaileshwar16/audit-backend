"""
generate_training_data.py
Generates a realistic labeled expense dataset for training ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

POLICY_LIMITS = {
    "Travel": 15000, "Food": 2000, "Hotel": 8000,
    "Fuel": 3000, "Office Supplies": 5000, "Entertainment": 4000,
}
CATEGORIES    = list(POLICY_LIMITS.keys())
DEPARTMENTS   = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations"]
PAYMENT_MODES = ["Credit Card", "UPI", "Cash", "Debit Card"]
VENDORS       = ["MakeMyTrip","Ola","Swiggy","Zomato","Amazon",
                 "OYO Rooms","Uber","IRCTC","Flipkart","Dmart"]

def generate_employee_profiles(n=50):
    profiles = []
    for i in range(n):
        profiles.append({
            "emp_id": f"E{i+1:03d}",
            "dept": np.random.choice(DEPARTMENTS),
            "avg_monthly_spend": np.random.randint(2000, 10000),
            "is_fraud_employee": np.random.rand() < 0.15,
        })
    return profiles

def generate_expense(emp, expense_id, ref_date):
    cat   = np.random.choice(CATEGORIES)
    limit = POLICY_LIMITS[cat]
    fraud = emp["is_fraud_employee"]

    over_limit       = fraud and np.random.rand() < 0.55
    is_duplicate     = fraud and np.random.rand() < 0.30
    no_receipt       = fraud and np.random.rand() < 0.40
    receipt_mismatch = fraud and np.random.rand() < 0.25

    if over_limit:
        amount = round(limit * np.random.uniform(1.1, 2.0), 2)
    elif fraud:
        amount = round(limit * np.random.uniform(0.5, 1.05), 2)
    else:
        amount = round(limit * np.random.uniform(0.1, 0.85), 2)

    days_back    = np.random.randint(0, 90)
    expense_date = ref_date - timedelta(days=days_back)
    if fraud and np.random.rand() < 0.45:
        while expense_date.weekday() < 5:
            expense_date -= timedelta(days=1)

    is_weekend           = expense_date.weekday() >= 5
    submissions_last_30d = np.random.randint(8, 20) if fraud else np.random.randint(1, 8)
    amount_vs_avg        = amount / max(emp["avg_monthly_spend"], 1)

    flags = []
    if amount > limit:             flags.append("Exceeds Policy Limit")
    if is_weekend:                 flags.append("Weekend Claim")
    if no_receipt:                 flags.append("Missing Receipt")
    if receipt_mismatch:           flags.append("Receipt Mismatch")
    if is_duplicate:               flags.append("Duplicate Claim")
    if submissions_last_30d > 12:  flags.append("Unusually Frequent")

    return {
        "expense_id":           f"EXP-{expense_id:05d}",
        "employee_id":          emp["emp_id"],       # aligned with fraud_engine
        "dept":                 emp["dept"],
        "avg_monthly":          emp["avg_monthly_spend"],
        "category":             cat,
        "amount":               amount,
        "policy_limit":         limit,
        "vendor":               np.random.choice(VENDORS),
        "date":                 expense_date.strftime("%Y-%m-%d"),
        "day_of_week":          expense_date.weekday(),
        "is_weekend":           int(is_weekend),
        "payment_mode":         np.random.choice(PAYMENT_MODES),
        "has_receipt":          int(not no_receipt),
        "receipt_mismatch":     int(receipt_mismatch),
        "is_duplicate":         int(is_duplicate),
        "amount_over_limit":    int(over_limit),     # aligned with fraud_engine
        "exceed_ratio":         round(amount / limit, 4),           # renamed
        "amount_vs_avg":        round(amount_vs_avg, 4),            # renamed
        "recent_submissions":   submissions_last_30d,               # renamed
        "flag_count":           len(flags),
        "flags":                "|".join(flags),
        "is_fraud":             1 if fraud and len(flags) >= 1 else 0,  # renamed
    }

def generate_dataset(n=3000):
    ref_date  = datetime(2025, 7, 15)
    employees = generate_employee_profiles(50)
    rows      = [generate_expense(employees[i % len(employees)], i+1, ref_date) for i in range(n)]
    df        = pd.DataFrame(rows)
    print(f"Generated {len(df)} expenses | Fraud rate: {df['is_fraud'].mean():.1%}")
    print(df['is_fraud'].value_counts())
    return df

if __name__ == "__main__":
    df = generate_dataset(3000)
    df.to_csv("data/expenses_training.csv", index=False)
    print("Saved to data/expenses_training.csv")
