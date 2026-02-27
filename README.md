# Expense Audit — ML Backend

Real ML fraud detection using **Random Forest** + **Isolation Forest** (scikit-learn).
No `Math.random()` — every score comes from a trained model.

---

## Architecture

```
expense-audit/
├── ml-backend/                  ← Python ML server (this folder)
│   ├── app.py                   ← Flask REST API entry point
│   ├── train.py                 ← Run once to train & save models
│   ├── requirements.txt
│   ├── data/
│   │   ├── generate_training_data.py   ← Synthetic labelled dataset
│   │   └── expenses_training.csv       ← Created by train.py
│   ├── models/
│   │   ├── fraud_engine.py             ← All 3 ML models
│   │   └── saved/engine.pkl            ← Saved after training
│   └── routes/
│       └── audit.py                    ← Flask Blueprint (API routes)
│
└── src/ (React frontend)
    └── services/
        └── auditApi.js          ← Replaces Math.random() with API calls
```

---

## ML Models Used

| Model | Algorithm | Purpose |
|---|---|---|
| FraudClassifier | Random Forest (200 trees) | Binary fraud prediction (0/1) |
| AnomalyDetector | Isolation Forest (global) | Detect overall spending outliers |
| EmployeeAnomalyDetector | Isolation Forest (per employee) | Detect per-person behavioural shifts |

### Features fed to Random Forest
| Feature | Description |
|---|---|
| `amount_over_limit` | Did amount exceed policy limit? (0/1) |
| `exceed_ratio` | amount ÷ policy_limit |
| `is_weekend` | Submitted on Saturday/Sunday? |
| `is_duplicate` | Same vendor+amount within 7 days? |
| `recent_submissions` | # submissions in past 14 days |
| `amount_vs_avg` | amount ÷ employee's avg monthly spend |
| `has_receipt` | Receipt attached? (0/1) |
| `day_of_week` | 0=Mon … 6=Sun |
| `category_enc` | Label-encoded expense category |
| `payment_mode_enc` | Label-encoded payment method |

### Risk Score Formula
```
risk_score = fraud_probability × 55        (Random Forest signal, 0–55)
           + anomaly_score × 0.30          (Isolation Forest,     0–30)
           + min(flag_count × 12, 36)      (Rule violations,      0–36)
```

---

## Setup & Run

### 1. Install Python dependencies
```bash
cd ml-backend
pip install -r requirements.txt
```

### 2. Train the models (run once)
```bash
python train.py
```
This will:
- Generate ~800 labelled training records
- Train Random Forest classifier → prints classification report + ROC-AUC
- Train global Isolation Forest
- Train per-employee Isolation Forest for each employee
- Save everything to `models/saved/engine.pkl`
- Run a smoke test with a fraud and clean expense

### 3. Start the API server
```bash
python app.py
# → Running on http://localhost:5000
```

### 4. Connect the React frontend
Copy these two files into your React project:
- `frontend_integration/auditApi.js`  → `src/services/auditApi.js`
- `frontend_integration/SubmitExpense.jsx` → `src/pages/SubmitExpense.jsx`

---

## API Endpoints

### `POST /api/audit/predict`
Score a single expense.
```json
{
  "employee_id":        "E002",
  "category":           "Travel",
  "amount":             24000,
  "vendor":             "MakeMyTrip",
  "date":               "2025-07-12",
  "payment_mode":       "Cash",
  "has_receipt":        false,
  "recent_submissions": 12,
  "avg_monthly":        7800,
  "is_duplicate":       true
}
```
Response:
```json
{
  "success": true,
  "result": {
    "fraud_probability": 0.91,
    "anomaly_score":     78.4,
    "risk_score":        89,
    "status":            "High Risk",
    "flags": [
      "Exceeds Policy Limit",
      "Receipt Missing",
      "Duplicate Claim",
      "Unusually Frequent Submissions"
    ],
    "feature_values": {
      "exceed_ratio":       1.6,
      "amount_vs_avg":      0.51,
      "recent_submissions": 12,
      "is_weekend":         true,
      "is_duplicate":       true,
      "has_receipt":        false
    }
  }
}
```

### `POST /api/audit/predict-batch`
Score multiple expenses at once.
```json
{ "expenses": [ <expense>, <expense>, … ] }
```

### `POST /api/audit/employee-profile`
Get employee-level anomaly score.
```json
{
  "employee_id": "E002",
  "scored_expenses": [ <result>, <result>, … ]
}
```

### `GET /api/audit/health`
Check if models are loaded.

---

## What replaced Math.random()

| Was random | Now ML |
|---|---|
| `Math.random() * 25` for base risk | `fraud_probability × 55` from Random Forest |
| `flags.length * random` for risk tiers | Isolation Forest anomaly score |
| `anomalyScore = flagCount / empExp.length * 150` | Per-employee Isolation Forest |
| `hasReceipt: Math.random() > 0.15` | Actual form field (user uploads receipt) |
| `is_duplicate: Math.random() > 0.5` | DB query: same vendor+amount within 7 days |
