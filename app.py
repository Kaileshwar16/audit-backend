"""
app.py  —  Entry point for the Expense Audit ML Backend
Run: python app.py
"""
from flask import Flask
from flask_cors import CORS
from routes.audit import audit_bp

app = Flask(__name__)
CORS(app)   # allows React frontend (localhost:5173) to call this API

app.register_blueprint(audit_bp, url_prefix="/api/audit")

@app.route("/")
def index():
    return {
        "service": "Expense Audit ML Backend",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/audit/predict":          "Score a single expense",
            "POST /api/audit/predict-batch":    "Score multiple expenses",
            "POST /api/audit/employee-profile": "Get employee risk profile",
            "GET  /api/audit/health":           "Model health check",
        }
    }

if __name__ == "__main__":
    print("Starting Expense Audit ML Backend on http://localhost:5000")
    print("Make sure you have run 'python train.py' first!\n")
    app.run(debug=True, port=5000)
