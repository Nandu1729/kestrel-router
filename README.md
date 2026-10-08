# Kestrel Home — Service-Request Router

Automated service-request routing for Kestrel Home Appliances.  
Replaces the Rs 3.2 lakh/year vendor routing bot with a locally-trained, zero-cost classifier.

## Quick Start (Clean Machine)

The pre-trained model is already included in `models/`, so the service runs immediately out of the box:

```bash
# 1. Clone and enter the directory
git clone https://github.com/Nandu1729/kestrel-router.git
cd kestrel-router

# 2. Create and activate a virtual environment (Python 3.9+)
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# 3. Install lightweight dependencies (scikit-learn, Flask, pandas)
pip install -r requirements.txt

# 4. Start the web service
PORT=8080 python app.py
# → Open http://localhost:8080 in your browser
```

> **Data Privacy Notice (Ops Policy §10):**  
> Per §10 of Kestrel's operations policy, raw customer data files are kept local and excluded from the public Git repository. The pre-trained model artifacts are tracked in `models/` so the web service boots and infers instantly. If you have the Kestrel data pack, place the CSV files in `data/` to re-run `python train_model.py` or `python predict.py`.

---

## What It Does

- **Classifies** incoming service requests into one of 7 teams
- **API endpoint**: `POST /api/predict` — takes a JSON record, returns the team, confidence score, and human-readable reasons
- **Web UI**: A clean single-screen UI at `http://localhost:8080` that calls the endpoint
- **Zero paid API keys needed** — runs entirely on scikit-learn (TF-IDF + calibrated LinearSVC)

---

## Project Structure

```
kestrel-router/
├── app.py                  # Flask web service (API + UI, CORS enabled)
├── train_model.py          # Model training pipeline (LinearSVC, 95.5% CV)
├── predict.py              # Batch prediction generator (<2s runtime)
├── predictions.csv         # 2,178 test predictions matching sample submission format
├── requirements.txt        # Python dependencies (no paid keys)
├── README.md               # Quick start & documentation
├── models/                 # Pre-trained models (included in repo)
│   ├── router_model.pkl    # Primary bot-matching model (95.5% accuracy)
│   ├── router_model_final_team.pkl  # Phase 2 true-resolution model
│   └── preprocessor.pkl    # Feature extraction & vectorizer
├── evidence/               # Validation reports and error analysis
│   ├── validation_report.py
│   └── validation_report.txt
├── memo_to_ritu.md         # 1-page business executive memo
├── submission-form.md      # Completed 13-question submission form
├── templates/
│   └── index.html          # Web UI
└── data/                   # (Excluded from Git per ops-policy §10)
    ├── train.csv
    ├── test_unlabelled.csv
    ├── resolution_log.csv
    ├── teams.csv
    ├── sample_submission.csv
    └── ops-policy.pdf
```

---

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
  "confidence": 0.98,
  "reasons": [
    "Fault indicators detected: blank",
    "Product (Air Fryer) appears to need technician attention",
    "Team handles: Product faults, breakdowns, error codes, noise, leaks — anything needing a technician."
  ],
  "team_description": "Product faults, breakdowns, error codes, noise, leaks — anything needing a technician."
}
```

---

## Graceful Degradation (Without Model)

If model files are removed or missing, the service **starts and fails politely**:
- The UI displays an informational warning banner
- The `/api/predict` endpoint returns HTTP 503 with a clear message
- The `/api/health` endpoint reports `"status": "degraded"`

---

## Cost Analysis

- **Per-prediction cost:** **Rs 0.00** (scikit-learn inference runs locally in <1ms)
- **Monthly cost at Kestrel volume (700 requests/month):** **Rs 0**
- **Annual savings:** **Rs 3,20,000 / year** by replacing the vendor bot license.

---

## Key Technical Decisions

1. **Target Selection:** The primary model is trained on `team_label` to satisfy Ritu's mandate of matching the 18 months of historical routing labels at 90%+ (**achieved 95.48% 5-fold CV accuracy**). A secondary model trained on `final_team` is preserved for Phase 2.
2. **Team Normalization:** Mapped 9 historical name variants into the 7 canonical teams (`Installations` → `Installs & Demo`, `Consumables` → `Filters & Consumables`).
3. **Data Sanitation:** Cleaned garbled Unicode from legacy Zoho records and avoided Zoho UTC timestamps per ops policy §9.
4. **Confidence Scoring:** Requests below 70% confidence (~4.6% of volume) can be flagged for manual review, preventing misrouting on vague requests like *"please call me"*.
