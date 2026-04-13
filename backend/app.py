"""
============================================================
  Edge-AI Fire Hazard Detection System
  Backend API — Flask

  Author  : [Your Name]
  Project : Academic Minor Project

  ENDPOINTS:
  POST /api/predict     → Receive sensor data, return ML prediction
  GET  /api/status      → System health check
  GET  /api/history     → Last N predictions
  GET  /api/stats       → Aggregate stats

  INSTALL:
  pip install flask flask-cors joblib scikit-learn numpy pandas

  RUN:
  python app.py
  → Server starts at http://0.0.0.0:5000
============================================================
"""

from flask          import Flask, request, jsonify
from flask_cors     import CORS
from datetime       import datetime
import numpy        as np
import joblib
import os
import json

# ============================================================
#  APP INIT
# ============================================================
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # Allow ESP32 and dashboard to call this API
from flask import send_from_directory

# ============================================================
#  LOAD ML MODEL
# ============================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "../model/fire_hazard_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "../model/scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "../model/label_encoder.pkl")

print("[API] Loading ML model...")
try:
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    le      = joblib.load(ENCODER_PATH)
    print("[API] Model loaded successfully [OK]")
    MODEL_LOADED = True
except Exception as e:
    print(f"[API] WARNING: Could not load model — {e}")
    print("[API] Falling back to rule-based prediction.")
    MODEL_LOADED = False

# ============================================================
#  IN-MEMORY LOG (stores last 200 predictions)
# ============================================================
MAX_HISTORY = 200
prediction_history = []

# --- Add one initial log entry so dashboard isn't empty on load ---
prediction_history.append({
    "timestamp": datetime.now().isoformat(),
    "device_id": "SYSTEM",
    "inputs": {"temperature": 0.0, "humidity": 0.0, "gas_level": 0.0},
    "prediction": {"label": "SAFE", "risk_score": 0.0, "confidence": 100.0},
    "alert": False
})

# ============================================================
#  HELPER: FEATURE ENGINEERING
#  Must match train_model.py exactly!
# ============================================================
def build_features(temperature, humidity, gas_level):
    temp_gas_index = temperature * gas_level / 1000
    hum_inverse    = 100 - humidity
    combined_risk  = (temperature / 100) + (hum_inverse / 100) + (gas_level / 1000)
    return np.array([[temperature, humidity, gas_level,
                      temp_gas_index, hum_inverse, combined_risk]])

# ============================================================
#  HELPER: RULE-BASED FALLBACK (if model not loaded)
# ============================================================
def rule_based_predict(temperature, humidity, gas_level):
    score = 0.0
    if temperature > 70:       score += 0.45
    elif temperature > 50:     score += 0.25
    elif temperature > 35:     score += 0.10

    if humidity < 20:          score += 0.20
    elif humidity < 35:        score += 0.10

    if gas_level > 500:        score += 0.40
    elif gas_level > 250:      score += 0.20
    elif gas_level > 100:      score += 0.05

    score = max(0.0, min(1.0, score))

    if score < 0.35:   label, conf = "SAFE",    0.90
    elif score < 0.65: label, conf = "WARNING", 0.85
    else:              label, conf = "FIRE",     0.93

    return label, score, conf

# ============================================================
#  ROUTE: POST /api/predict
#  Called by ESP32 every few seconds
# ============================================================
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # --- Validate required fields ---
        required = ["temperature", "humidity", "gas_level"]
        missing  = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        temperature = float(data["temperature"])
        humidity    = float(data["humidity"])
        gas_level   = float(data["gas_level"])
        device_id   = data.get("device_id", "UNKNOWN")

        # --- Sanity bounds check ---
        if not (0 <= temperature <= 150):
            return jsonify({"error": "Temperature out of range (0-150°C)"}), 400
        if not (0 <= humidity <= 100):
            return jsonify({"error": "Humidity out of range (0-100%)"}), 400
        if not (0 <= gas_level <= 1500):
            return jsonify({"error": "Gas level out of range (0-1500 ppm)"}), 400

        # --- Predict (Prioritize ESP32 evaluation if available) ---
        if "risk_label" in data:
            pred_label = data["risk_label"]
            risk_score = data.get("risk_score", 0.0)
            confidence = data.get("confidence", 0.90)
        elif MODEL_LOADED:
            features        = build_features(temperature, humidity, gas_level)
            features_scaled = scaler.transform(features)
            pred_enc        = model.predict(features_scaled)[0]
            pred_label      = le.inverse_transform([pred_enc])[0]
            probas          = model.predict_proba(features_scaled)[0]
            confidence      = float(probas.max())
            risk_score      = 1.0 - probas[le.transform(["SAFE"])[0]]
        else:
            pred_label, risk_score, confidence = rule_based_predict(
                temperature, humidity, gas_level
            )

        # --- Build response ---
        timestamp = datetime.now().isoformat()
        result = {
            "timestamp"   : timestamp,
            "device_id"   : device_id,
            "inputs": {
                "temperature" : temperature,
                "humidity"    : humidity,
                "gas_level"   : gas_level
            },
            "prediction": {
                "label"      : pred_label,
                "risk_score" : round(float(risk_score), 4),
                "confidence" : round(confidence * 100, 2)
            },
            "alert": pred_label == "FIRE"
        }

        # --- Store in history ---
        prediction_history.append(result)
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)

        # --- Log to console ---
        print(f"[{timestamp}] Device={device_id} | "
              f"T={temperature}C H={humidity}% G={gas_level}ppm "
              f"-> {pred_label} ({confidence*100:.1f}%)")

        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": f"Invalid value: {str(ve)}"}), 400
    except Exception as e:
        print(f"[ERROR] /api/predict — {e}")
        return jsonify({"error": "Internal server error"}), 500


# ============================================================
#  ROUTE: GET /api/status
#  Health check — ESP32 or dashboard can ping this
# ============================================================
@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status"        : "online",
        "model_loaded"  : MODEL_LOADED,
        "total_readings": len(prediction_history),
        "server_time"   : datetime.now().isoformat(),
        "version"       : "1.0.0"
    }), 200


# ============================================================
#  ROUTE: GET /api/history?limit=20
#  Returns last N predictions
# ============================================================
@app.route("/api/history", methods=["GET"])
def history():
    try:
        limit = int(request.args.get("limit", 20))
        limit = min(limit, MAX_HISTORY)
        data  = prediction_history[-limit:][::-1]  # newest first
        return jsonify({
            "count"  : len(data),
            "data"   : data
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/latest")
def latest():
    if not prediction_history:
        return jsonify({"error": "No data yet"}), 404
    return jsonify(prediction_history[-1])


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ============================================================
#  ROUTE: GET /api/stats
#  Aggregate stats for dashboard charts
# ============================================================
@app.route("/api/stats", methods=["GET"])
def stats():
    if not prediction_history:
        return jsonify({"message": "No data yet"}), 200

    labels = [p["prediction"]["label"] for p in prediction_history]
    temps  = [p["inputs"]["temperature"] for p in prediction_history]
    hums   = [p["inputs"]["humidity"]    for p in prediction_history]
    gases  = [p["inputs"]["gas_level"]   for p in prediction_history]

    return jsonify({
        "total_readings"  : len(prediction_history),
        "label_counts": {
            "SAFE"    : labels.count("SAFE"),
            "WARNING" : labels.count("WARNING"),
            "FIRE"    : labels.count("FIRE")
        },
        "averages": {
            "temperature" : round(np.mean(temps), 2),
            "humidity"    : round(np.mean(hums),  2),
            "gas_level"   : round(np.mean(gases), 2)
        },
        "latest" : prediction_history[-1] if prediction_history else None
    }), 200


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Edge-AI Fire Hazard Detection — Flask API")
    print("=" * 50)
    print("  Endpoints:")
    print("  POST http://0.0.0.0:5000/api/predict")
    print("  GET  http://0.0.0.0:5000/api/status")
    print("  GET  http://0.0.0.0:5000/api/history")
    print("  GET  http://0.0.0.0:5000/api/stats")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
