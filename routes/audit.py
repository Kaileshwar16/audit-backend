"""
routes/audit.py
Flask Blueprint exposing the ML fraud engine via REST API.
"""
from flask import Blueprint, request, jsonify
from models.fraud_engine import FraudEngine
from models.receipt_ocr import ReceiptVerifier
import traceback
import os
import uuid

audit_bp  = Blueprint("audit", __name__)
_engine   = None
_verifier = ReceiptVerifier()

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_engine() -> FraudEngine:
    global _engine
    if _engine is None:
        _engine = FraudEngine()
        _engine._load()
    return _engine


# ── POST /api/audit/predict ───────────────────────────────────────────────────
@audit_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        required = ["employee_id", "category", "amount", "date"]
        missing  = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        result = get_engine().predict(data)
        return jsonify({"success": True, "result": result}), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e), "hint": "Run train.py first"}), 503
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── POST /api/audit/predict-batch ─────────────────────────────────────────────
@audit_bp.route("/predict-batch", methods=["POST"])
def predict_batch():
    try:
        data     = request.get_json(force=True)
        expenses = data.get("expenses", [])
        if not expenses:
            return jsonify({"error": "expenses list is empty"}), 400

        engine  = get_engine()
        results = [engine.predict(exp) for exp in expenses]
        return jsonify({"success": True, "results": results}), 200

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── POST /api/audit/employee-profile ──────────────────────────────────────────
@audit_bp.route("/employee-profile", methods=["POST"])
def employee_profile():
    try:
        data     = request.get_json(force=True)
        emp_id   = data.get("employee_id", "")
        expenses = data.get("scored_expenses", [])

        profile = get_engine().predict_employee_profile(emp_id, expenses)
        return jsonify({"success": True, "profile": profile}), 200

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── POST /api/audit/verify-receipt ────────────────────────────────────────────
@audit_bp.route("/verify-receipt", methods=["POST"])
def verify_receipt():
    """
    Upload a receipt image or PDF and verify it against the claimed expense.

    Multipart form fields:
      receipt  (file)    — image (jpg/png/webp) or PDF
      amount   (float)   — claimed amount
      vendor   (string)  — claimed vendor name
      date     (string)  — claimed date YYYY-MM-DD

    Returns:
      {
        verified:    bool,
        confidence:  float (0-100),
        extracted:   { amount, vendor, date },
        mismatches:  [ "Amount mismatch: ..." ],
        ocr_text:    "raw OCR output"
      }
    """
    try:
        if "receipt" not in request.files:
            return jsonify({"error": "No receipt file uploaded"}), 400

        file = request.files["receipt"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".pdf"}:
            return jsonify({"error": f"Unsupported file type '{ext}'. Use jpg, png, webp, or pdf."}), 400

        filename  = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        claimed = {
            "amount": float(request.form.get("amount", 0) or 0) or None,
            "vendor": request.form.get("vendor", "").strip() or None,
            "date":   request.form.get("date",   "").strip() or None,
        }

        result = _verifier.verify(save_path, claimed)

        try:
            os.remove(save_path)
        except OSError:
            pass

        return jsonify({"success": True, "result": result}), 200

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── GET /api/audit/health ──────────────────────────────────────────────────────
@audit_bp.route("/health", methods=["GET"])
def health():
    try:
        engine = get_engine()
        return jsonify({
            "status":      "ok",
            "model_ready": engine.is_trained,
            "ocr_ready":   True,
            "employees_with_personal_model": list(engine.emp_models.keys()),
        }), 200
    except FileNotFoundError:
        return jsonify({"status": "model_not_trained", "hint": "Run train.py"}), 503
