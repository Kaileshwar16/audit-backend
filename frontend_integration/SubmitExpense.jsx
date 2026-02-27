/**
 * src/pages/SubmitExpense.jsx  (ML-integrated version)
 *
 * Changes from the random version:
 *   - Calls predictExpense() from auditApi.js instead of using Math.random()
 *   - Risk score, fraud probability, flags and status all come from Python ML backend
 *   - Shows fraud_probability and anomaly_score in result card
 */
import { useState } from "react";
import { G, css } from "../utils/theme";
import { fmt } from "../utils/helpers";
import { EMPLOYEES, POLICY_LIMITS, PAYMENT_MODES } from "../data/mockData";
import RiskBar from "../components/RiskBar";
import { predictExpense } from "../services/auditApi";   // ← ML API

const DEFAULT_FORM = {
  empId:       "",
  category:    "Travel",
  amount:      "",
  vendor:      "",
  date:        "",
  paymentMode: "Credit Card",
  location:    "",
};

export default function SubmitExpense({ onSubmit }) {
  const [form,    setForm]    = useState(DEFAULT_FORM);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const runAudit = async () => {
    if (!form.empId || !form.amount || !form.vendor || !form.date || !form.location) return;

    setLoading(true);
    setError(null);

    try {
      const emp = EMPLOYEES.find((e) => e.id === form.empId) || EMPLOYEES[0];

      // Build the payload for the ML backend
      const payload = {
        employee_id:        form.empId,
        category:           form.category,
        amount:             Number(form.amount),
        vendor:             form.vendor,
        date:               form.date,
        payment_mode:       form.paymentMode,
        has_receipt:        true,          // user uploaded receipt
        recent_submissions: 0,             // could track this in real app
        avg_monthly:        emp.avgMonthly,
        is_duplicate:       false,         // backend/DB would check this
      };

      // ── Call the ML backend ──────────────────────────────────────────────
      const mlResult = await predictExpense(payload);

      const newExp = {
        id:          `EXP-${Math.floor(Math.random() * 9000 + 1000)}`,
        empName:     emp.name,
        dept:        emp.dept,
        ...form,
        amount:      Number(form.amount),
        // All of these now come from real ML — no Math.random()!
        flags:            mlResult.flags,
        risk:             mlResult.risk_score,
        status:           mlResult.status,
        fraudProbability: mlResult.fraud_probability,
        anomalyScore:     mlResult.anomaly_score,
        featureValues:    mlResult.feature_values,
        hasReceipt:       true,
      };

      setResult(newExp);
      onSubmit(newExp);

    } catch (err) {
      setError(`ML backend error: ${err.message}. Is the Python server running?`);
    } finally {
      setLoading(false);
    }
  };

  const statusColor = (s) =>
    s === "Approved"     ? G.success :
    s === "Needs Review" ? G.warning :
    G.danger;

  return (
    <div style={{ maxWidth: 700, margin: "0 auto" }}>
      <div style={{ marginBottom: "2rem" }}>
        <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: "0.05em", marginBottom: 4 }}>
          SUBMIT EXPENSE CLAIM
        </div>
        <div style={{ fontSize: 12, color: G.muted }}>
          Powered by Random Forest + Isolation Forest · No hardcoded rules
        </div>
      </div>

      {error && (
        <div style={{ background: `${G.danger}15`, border: `1px solid ${G.danger}44`, borderRadius: 6, padding: "1rem", marginBottom: "1rem", fontSize: 13, color: G.danger }}>
          ⚠ {error}
        </div>
      )}

      {result ? (
        /* ── ML Result Card ── */
        <div style={css.card}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: "1.5rem" }}>
            <div>
              <div style={{ fontSize: 12, color: G.muted }}>ML Audit Complete</div>
              <div style={{ fontSize: 22, fontWeight: 800, color: G.accent }}>{result.id}</div>
            </div>
            <span style={{
              padding: "6px 14px", borderRadius: 4, fontWeight: 700, fontSize: 13,
              background: statusColor(result.status) + "22",
              color: statusColor(result.status),
              border: `1px solid ${statusColor(result.status)}44`,
            }}>
              {result.status}
            </span>
          </div>

          {/* Risk Score */}
          <div style={{ marginBottom: "1.25rem" }}>
            <div style={css.label}>Risk Score (ML)</div>
            <RiskBar score={result.risk} />
          </div>

          {/* ML Signals */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1.5rem" }}>
            <div style={{ background: G.surface2, borderRadius: 6, padding: "0.75rem" }}>
              <div style={css.label}>Fraud Probability</div>
              <div style={{ fontSize: 20, fontWeight: 800, color: result.fraudProbability > 0.5 ? G.danger : G.success }}>
                {(result.fraudProbability * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 11, color: G.muted }}>Random Forest</div>
            </div>
            <div style={{ background: G.surface2, borderRadius: 6, padding: "0.75rem" }}>
              <div style={css.label}>Anomaly Score</div>
              <div style={{ fontSize: 20, fontWeight: 800, color: result.anomalyScore > 50 ? G.warning : G.success }}>
                {result.anomalyScore}
              </div>
              <div style={{ fontSize: 11, color: G.muted }}>Isolation Forest</div>
            </div>
          </div>

          {/* Feature Explanation */}
          {result.featureValues && (
            <div style={{ marginBottom: "1.5rem" }}>
              <div style={css.label}>Model Feature Values</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.5rem", marginTop: 8 }}>
                {Object.entries(result.featureValues).map(([k, v]) => (
                  <div key={k} style={{ background: G.surface2, borderRadius: 4, padding: "6px 10px" }}>
                    <div style={{ fontSize: 10, color: G.muted, textTransform: "uppercase" }}>{k.replace(/_/g, " ")}</div>
                    <div style={{ fontSize: 13, fontWeight: 700, marginTop: 2 }}>
                      {typeof v === "boolean" ? (v ? "Yes" : "No") : v}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Flags */}
          {result.flags.length > 0 ? (
            <div style={{ marginBottom: "1.5rem" }}>
              <div style={css.label}>Audit Flags</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>
                {result.flags.map((f) => <span key={f} style={css.tag(G.danger)}>{f}</span>)}
              </div>
            </div>
          ) : (
            <div style={{ ...css.tag(G.success), marginBottom: "1.5rem", fontSize: 12 }}>
              ✓ No policy violations detected
            </div>
          )}

          <div style={{ background: G.surface2, borderRadius: 6, padding: "1rem", fontSize: 12, color: G.muted }}>
            Policy limit for <strong style={{ color: G.text }}>{result.category}</strong>:{" "}
            <strong style={{ color: G.text }}>{fmt(POLICY_LIMITS[result.category])}</strong>
            {" "}· Your claim:{" "}
            <strong style={{ color: result.amount > POLICY_LIMITS[result.category] ? G.danger : G.success }}>
              {fmt(result.amount)}
            </strong>
          </div>

          <button
            style={{ ...css.btn(), marginTop: "1.5rem" }}
            onClick={() => { setResult(null); setForm(DEFAULT_FORM); }}
          >
            Submit Another
          </button>
        </div>
      ) : (
        /* ── Form ── */
        <div style={css.card}>
          <div style={css.grid2}>
            <div>
              <label style={css.label}>Employee ID</label>
              <select style={css.input} value={form.empId} onChange={set("empId")}>
                <option value="">Select Employee</option>
                {EMPLOYEES.map((e) => <option key={e.id} value={e.id}>{e.id} — {e.name}</option>)}
              </select>
            </div>
            <div>
              <label style={css.label}>Category</label>
              <select style={css.input} value={form.category} onChange={set("category")}>
                {Object.keys(POLICY_LIMITS).map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label style={css.label}>Amount (₹)</label>
              <input style={css.input} type="number" placeholder="0" value={form.amount} onChange={set("amount")} />
            </div>
            <div>
              <label style={css.label}>Vendor</label>
              <input style={css.input} placeholder="Vendor name" value={form.vendor} onChange={set("vendor")} />
            </div>
            <div>
              <label style={css.label}>Date</label>
              <input style={css.input} type="date" value={form.date} onChange={set("date")} />
            </div>
            <div>
              <label style={css.label}>Payment Mode</label>
              <select style={css.input} value={form.paymentMode} onChange={set("paymentMode")}>
                {PAYMENT_MODES.map((p) => <option key={p}>{p}</option>)}
              </select>
            </div>
            <div style={{ gridColumn: "1 / -1" }}>
              <label style={css.label}>Location</label>
              <input style={css.input} placeholder="City / Location" value={form.location} onChange={set("location")} />
            </div>
          </div>

          <div style={{ marginTop: "1.5rem", background: G.surface2, borderRadius: 6, padding: "1rem", fontSize: 12, color: G.muted }}>
            Policy Limits: {Object.entries(POLICY_LIMITS).map(([k, v]) => `${k}: ${fmt(v)}`).join(" · ")}
          </div>

          <button
            style={{ ...css.btn(), marginTop: "1.5rem", opacity: loading ? 0.6 : 1 }}
            onClick={runAudit}
            disabled={loading}
          >
            {loading ? "RUNNING ML AUDIT…" : "SUBMIT & RUN ML AUDIT"}
          </button>
        </div>
      )}
    </div>
  );
}
