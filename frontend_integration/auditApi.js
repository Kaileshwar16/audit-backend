/**
 * src/services/auditApi.js
 *
 * Replaces all Math.random() risk/fraud logic in the frontend.
 * Every expense now gets scored by the Python ML backend.
 */

const BASE_URL = "http://localhost:5000/api/audit";

// ── Single expense prediction ─────────────────────────────────────────────────
export async function predictExpense(expense) {
  /**
   * expense shape:
   *   employee_id, category, amount, vendor, date,
   *   payment_mode, has_receipt, recent_submissions,
   *   avg_monthly, is_duplicate
   *
   * Returns:
   *   { fraud_probability, anomaly_score, risk_score, status, flags, feature_values }
   */
  const res = await fetch(`${BASE_URL}/predict`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(expense),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  const data = await res.json();
  return data.result;
}


// ── Batch prediction (for loading all expenses at once) ───────────────────────
export async function predictBatch(expenses) {
  /**
   * expenses: array of expense objects (same shape as predictExpense)
   * Returns:  array of result objects in same order
   */
  const res = await fetch(`${BASE_URL}/predict-batch`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ expenses }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  const data = await res.json();
  return data.results;
}


// ── Employee risk profile ─────────────────────────────────────────────────────
export async function getEmployeeProfile(employeeId, scoredExpenses) {
  /**
   * scoredExpenses: array of results already returned by predictExpense()
   * Returns: { anomaly_score, is_anomaly, avg_risk_score, flag_rate, top_flags, … }
   */
  const res = await fetch(`${BASE_URL}/employee-profile`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      employee_id:     employeeId,
      scored_expenses: scoredExpenses,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  const data = await res.json();
  return data.profile;
}


// ── Health check ──────────────────────────────────────────────────────────────
export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}
