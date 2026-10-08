"""
Kestrel Home — Service-Request Router
A Flask web service that classifies incoming service requests
into the correct team queue using a trained ML model.
"""

import os
import re
import logging
from datetime import datetime

import numpy as np
import scipy.sparse as sp
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model loading ──────────────────────────────────────────────────────────
MODEL = None
PREPROCESSOR = None
MODEL_LOAD_ERROR = None

TEAM_DESCRIPTIONS = {
    "Installs & Demo": "New-product installation, demo and wall-mounting visits.",
    "Repairs": "Product faults, breakdowns, error codes, noise, leaks — anything needing a technician.",
    "Filters & Consumables": "Filters, candles, membranes, jars, brushes, blades, AMC kits — selling and fitting spares.",
    "Billing": "Invoices, GST, double charges, refunds of payments, EMI conversion, coupons. Only when the problem IS the payment.",
    "Returns & Replacement": "Damaged, wrong or incomplete deliveries; returns and exchanges within the return window.",
    "Warranty Claims": "Warranty and Kestrel Shield registration, coverage questions, claim status.",
    "Product Advice": "Pre- and post-purchase usage questions. No fault reported.",
}

VALID_TEAMS = list(TEAM_DESCRIPTIONS.keys())
VALID_CHANNELS = ["ivr", "chat", "whatsapp", "email"]
VALID_PRODUCTS = [
    "Air Fryer", "Ceiling Fan", "Induction Cooktop",
    "Mixer Grinder", "Robot Vacuum", "Room Heater", "Water Purifier"
]
VALID_WARRANTY = ["in_warranty", "out_of_warranty", "shield"]


def load_model():
    """Try to load the trained model. Fail gracefully if missing."""
    global MODEL, PREPROCESSOR, MODEL_LOAD_ERROR
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "router_model.pkl")
    preproc_path = os.path.join(model_dir, "preprocessor.pkl")

    if not os.path.exists(model_path) or not os.path.exists(preproc_path):
        MODEL_LOAD_ERROR = (
            "Model files not found. Please run 'python train_model.py' first "
            "to train the classifier."
        )
        logger.warning(MODEL_LOAD_ERROR)
        return

    try:
        import joblib
        MODEL = joblib.load(model_path)
        PREPROCESSOR = joblib.load(preproc_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        MODEL_LOAD_ERROR = f"Failed to load model: {e}"
        logger.error(MODEL_LOAD_ERROR)


load_model()


def clean_text(text):
    """Clean garbled Unicode and normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def extract_features(record):
    """
    Build the feature vector for a single record.
    Uses the SAME feature lists stored in the preprocessor to guarantee
    feature count matches the trained model.
    """
    text = clean_text(record.get("request_text", ""))

    # Load feature config from preprocessor (guaranteed to match training)
    valid_channels = PREPROCESSOR['valid_channels']
    valid_products = PREPROCESSOR['valid_products']
    valid_warranty = PREPROCESSOR['valid_warranty']
    keyword_features = PREPROCESSOR['keyword_features']

    # TF-IDF features
    tfidf = PREPROCESSOR['tfidf']
    tfidf_vec = tfidf.transform([text])

    # Channel one-hot
    channel = record.get("channel", "").lower()
    channel_feats = [1 if c == channel else 0 for c in valid_channels]

    # Product family one-hot
    product = record.get("product_family", "")
    product_feats = [1 if p == product else 0 for p in valid_products]

    # Warranty status one-hot
    warranty = record.get("warranty_status", "")
    warranty_feats = [1 if w == warranty else 0 for w in valid_warranty]

    # Keyword features
    kw_feats = [1 if kw in text else 0 for kw in keyword_features]

    # Combine structured features
    structured = np.array(
        channel_feats + product_feats + warranty_feats + kw_feats
    ).reshape(1, -1)

    # Combine all features
    combined = sp.hstack([tfidf_vec, sp.csr_matrix(structured)])
    return combined


def generate_reasons(record, predicted_team):
    """
    Generate human-readable reasons for the routing decision,
    suitable for a Kestrel employee to read.
    """
    text = clean_text(record.get("request_text", ""))
    product = record.get("product_family", "Unknown")
    warranty = record.get("warranty_status", "unknown").replace("_", " ")
    reasons = []

    if predicted_team == "Repairs":
        found = [kw for kw in ['error code', 'noise', 'leak', 'not working',
                                'broken', 'tripping', 'stopped working',
                                'blank', 'fault', 'smell', 'smoke']
                 if kw in text]
        if found:
            reasons.append(f"Fault indicators detected: {', '.join(found)}")
        reasons.append(f"Product ({product}) appears to need technician attention")

    elif predicted_team == "Installs & Demo":
        found = [kw for kw in ['install', 'demo', 'wall mount', 'mounting',
                                'delivered but not installed']
                 if kw in text]
        if found:
            reasons.append(f"Installation-related terms: {', '.join(found)}")
        reasons.append(f"Request relates to setting up or demonstrating {product}")

    elif predicted_team == "Filters & Consumables":
        found = [kw for kw in ['filter', 'spare', 'blade', 'jar', 'membrane',
                                'candle', 'brush', 'amc', 'consumable']
                 if kw in text]
        if found:
            reasons.append(f"Spare/consumable items mentioned: {', '.join(found)}")
        reasons.append("Request is about purchasing or fitting replacement parts, not a fault")

    elif predicted_team == "Billing":
        found = [kw for kw in ['invoice', 'refund', 'emi', 'coupon', 'gst',
                                'double charge', 'payment', 'receipt']
                 if kw in text]
        if found:
            reasons.append(f"Payment-related issue: {', '.join(found)}")
        reasons.append("The problem is the payment itself, not the product")

    elif predicted_team == "Returns & Replacement":
        found = [kw for kw in ['return', 'cancel', 'exchange', 'missing parts',
                                'damaged', 'wrong product', 'pickup']
                 if kw in text]
        if found:
            reasons.append(f"Return/replacement indicators: {', '.join(found)}")
        reasons.append("Request involves delivery issues or product exchange")

    elif predicted_team == "Warranty Claims":
        found = [kw for kw in ['warranty', 'claim', 'shield', 'coverage',
                                'registration', 'warranty card']
                 if kw in text]
        if found:
            reasons.append(f"Warranty-related terms: {', '.join(found)}")
        reasons.append(f"Product warranty status: {warranty}")

    elif predicted_team == "Product Advice":
        found = [kw for kw in ['how to', 'best settings', 'recipe',
                                'power consumption', 'clean', 'temperature',
                                'usage']
                 if kw in text]
        if found:
            reasons.append(f"Usage/advice query: {', '.join(found)}")
        reasons.append("No fault reported — customer needs product guidance")

    if not reasons:
        reasons.append(f"Based on overall text analysis of the service request for {product}")

    reasons.append(f"Team handles: {TEAM_DESCRIPTIONS.get(predicted_team, 'N/A')}")
    return reasons


# ── API Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main routing UI."""
    return render_template(
        "index.html",
        products=VALID_PRODUCTS,
        channels=VALID_CHANNELS,
        warranty_options=VALID_WARRANTY,
        model_loaded=MODEL is not None,
        model_error=MODEL_LOAD_ERROR,
    )


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "model_error": MODEL_LOAD_ERROR,
        "timestamp": datetime.now().isoformat(),
    })


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Takes a single service request record as JSON.
    Returns the predicted team and human-readable reasons.

    Request body:
    {
        "request_text": "my air fryer display has gone blank",
        "channel": "whatsapp",
        "product_family": "Air Fryer",
        "warranty_status": "in_warranty"
    }

    Response:
    {
        "team": "Repairs",
        "confidence": 0.87,
        "reasons": ["Fault indicators detected: blank", ...],
        "team_description": "Product faults, breakdowns, ..."
    }
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if MODEL is None:
        return jsonify({
            "error": "Model not loaded. " + (MODEL_LOAD_ERROR or ""),
            "suggestion": "Run 'python train_model.py' to train the model first."
        }), 503

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    if not data.get("request_text"):
        return jsonify({"error": "Field 'request_text' is required."}), 400

    try:
        features = extract_features(data)
        predicted_team = MODEL.predict(features)[0]

        # Get confidence
        confidence = None
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(features)[0]
            class_idx = list(MODEL.classes_).index(predicted_team)
            confidence = round(float(proba[class_idx]), 4)
        elif hasattr(MODEL, "decision_function"):
            decision = MODEL.decision_function(features)[0]
            exp_d = np.exp(decision - np.max(decision))
            proba = exp_d / exp_d.sum()
            class_idx = list(MODEL.classes_).index(predicted_team)
            confidence = round(float(proba[class_idx]), 4)

        reasons = generate_reasons(data, predicted_team)

        return jsonify({
            "team": predicted_team,
            "confidence": confidence,
            "reasons": reasons,
            "team_description": TEAM_DESCRIPTIONS.get(predicted_team, ""),
        })

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/teams", methods=["GET"])
def list_teams():
    """List all available teams and what they handle."""
    return jsonify({
        "teams": [
            {"name": name, "description": desc}
            for name, desc in TEAM_DESCRIPTIONS.items()
        ]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
