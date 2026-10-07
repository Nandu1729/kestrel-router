# Kestrel Home — Service-Request Router

Automated service-request routing for Kestrel Home Appliances.  
Replaces the Rs 3.2 lakh/year vendor routing bot with a locally-trained classifier.

## Quick Start (clean machine)

```bash
# 1. Clone and enter the directory
git clone <repo-url>
cd kestrel-router

# 2. Create a virtual environment (Python 3.9+)
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (~30 seconds)
python train_model.py

# 5. Generate predictions for the test set
python predict.py

# 6. Start the web service
PORT=8080 python app.py
# → Open http://localhost:8080 in your browser
```

## What It Does

- **Classifies** incoming service requests into one of 7 teams
- **API endpoint**: `POST /api/predict` — takes a JSON record, returns the team + human-readable reasons
- **Web UI**: A single screen at `/` that calls the endpoint and shows results
- **No paid API keys needed** — runs entirely on scikit-learn (TF-IDF + classifier)

## Project Structure

```
kestrel-router/
├── app.py                  # Flask web service (API + UI)
├── train_model.py          # Model training pipeline
├── predict.py              # Generate predictions.csv
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   ├── train.csv           # Training data with team_label
│   ├── test_unlabelled.csv # Test data (no labels)
│   ├── resolution_log.csv  # Where requests actually resolved
│   ├── teams.csv           # Team definitions and renames
│   ├── sample_submission.csv
│   └── ops-policy.pdf
├── models/                 # Trained model files (created by train_model.py)
│   ├── router_model.pkl
│   └── preprocessor.pkl
├── evidence/               # Validation reports and analysis
│   └── validation_report.txt
├── predictions.csv         # Output predictions
├── memo_to_ritu.md         # One-page business memo
└── templates/
    └── index.html          # Web UI
```

## API Usage

```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "request_text": "my air fryer display has gone blank",
    "channel": "whatsapp",
    "product_family": "Air Fryer",
    "warranty_status": "in_warranty"
  }'
```

Response:
```json
{
  "team": "Repairs",
  "confidence": 0.91,
  "reasons": [
    "Fault indicators detected: blank",
    "Product (Air Fryer) appears to need technician attention",
    "Team handles: Product faults, breakdowns, error codes, noise, leaks..."
  ],
  "team_description": "Product faults, breakdowns, error codes..."
}
```

## Without a Model

If the model files are missing, the service **starts and fails politely**:
- The UI shows a warning banner explaining the model isn't loaded
- The `/api/predict` endpoint returns HTTP 503 with a helpful message
- The `/api/health` endpoint reports `"status": "degraded"`

## Cost

**Zero.** No paid API calls. The model is a locally-trained scikit-learn classifier.  
Monthly run cost at 700 requests: Rs 0 (compute is negligible on any modern server).

## Technical Decisions

1. **Trained on `final_team`** (from resolution_log.csv) — where requests actually ended up — rather than `team_label` (the bot's initial routing which was wrong 23% of the time)
2. **Normalized team names** — `Installations` → `Installs & Demo`, `Consumables` → `Filters & Consumables`
3. **TF-IDF + keyword features + structured features** — robust, fast, interpretable
4. **Cleaned garbled Unicode** from legacy Zoho data
