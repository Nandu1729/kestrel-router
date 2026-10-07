"""
Kestrel Home — Model Training Pipeline
Trains a text classifier to route service requests to the correct team.

Strategy:
- PRIMARY MODEL: Trained on team_label (bot's labels) — this is what Ritu asked to match
  ("matches those labels at 90%"). Achieves ~95% CV accuracy against bot labels.
- EVIDENCE: Also shows what a final_team model would achieve — for the memo to Ritu.

Team name normalization: Installations→Installs & Demo, Consumables→Filters & Consumables
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Team name normalization ─────────────────────────────────────────────────
TEAM_RENAME = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables",
}

CANONICAL_TEAMS = [
    "Installs & Demo", "Repairs", "Filters & Consumables",
    "Billing", "Returns & Replacement", "Warranty Claims", "Product Advice",
]

VALID_CHANNELS = ["ivr", "chat", "whatsapp", "email"]
VALID_PRODUCTS = [
    "Air Fryer", "Ceiling Fan", "Induction Cooktop",
    "Mixer Grinder", "Robot Vacuum", "Room Heater", "Water Purifier"
]
VALID_WARRANTY = ["in_warranty", "out_of_warranty", "shield"]

KEYWORD_FEATURES = [
    'install', 'demo', 'wall mount', 'mounting', 'delivered but not installed',
    'filter', 'spare', 'blade', 'jar', 'membrane', 'candle', 'brush', 'amc', 'consumable',
    'invoice', 'refund', 'emi', 'coupon', 'gst', 'double charge', 'payment', 'receipt',
    'return', 'cancel', 'exchange', 'missing parts', 'damaged', 'wrong product', 'pickup',
    'warranty', 'claim', 'shield', 'coverage', 'registration', 'warranty card',
    'error code', 'noise', 'leak', 'not working', 'broken', 'tripping',
    'stopped working', 'blank', 'fault', 'smell', 'smoke',
    'how to', 'best settings', 'recipe', 'power consumption', 'clean', 'temperature', 'usage',
    'call back', 'call me', 'please call',
]


def clean_text(text):
    """Clean garbled Unicode and normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def normalize_team(name):
    """Map old team names to current canonical names."""
    return TEAM_RENAME.get(name, name)


def build_structured_features(df):
    """Build channel + product + warranty + keyword features as a sparse matrix."""
    n = len(df)

    channel_mat = np.zeros((n, len(VALID_CHANNELS)))
    for i, ch in enumerate(df['channel'].values):
        if ch in VALID_CHANNELS:
            channel_mat[i, VALID_CHANNELS.index(ch)] = 1

    product_mat = np.zeros((n, len(VALID_PRODUCTS)))
    for i, p in enumerate(df['product_family'].values):
        if p in VALID_PRODUCTS:
            product_mat[i, VALID_PRODUCTS.index(p)] = 1

    warranty_mat = np.zeros((n, len(VALID_WARRANTY)))
    for i, w in enumerate(df['warranty_status'].values):
        if w in VALID_WARRANTY:
            warranty_mat[i, VALID_WARRANTY.index(w)] = 1

    texts = df['request_text_clean'].values
    keyword_mat = np.zeros((n, len(KEYWORD_FEATURES)))
    for i, text in enumerate(texts):
        for j, kw in enumerate(KEYWORD_FEATURES):
            if kw in text:
                keyword_mat[i, j] = 1

    structured = np.hstack([channel_mat, product_mat, warranty_mat, keyword_mat])
    return sp.csr_matrix(structured)


def main():
    print("=" * 70)
    print("KESTREL HOME — MODEL TRAINING PIPELINE")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    resolution = pd.read_csv(os.path.join(DATA_DIR, "resolution_log.csv"))
    print(f"  Train rows: {len(train_df)}")

    # ── Merge and prepare labels ─────────────────────────────────────────
    print("\n[2/7] Preparing labels...")
    merged = train_df.merge(
        resolution[['request_id', 'final_team', 'transfers']],
        on='request_id', how='left'
    )
    merged['label_bot'] = merged['team_label'].apply(normalize_team)
    merged['label_final'] = merged['final_team'].apply(normalize_team)

    misroute_rate = (merged['label_bot'] != merged['label_final']).mean()
    print(f"  Bot misroute rate: {misroute_rate:.1%}")

    print(f"\n  Bot-label distribution:")
    for team, count in merged['label_bot'].value_counts().sort_values(ascending=False).items():
        print(f"    {team:30s}  {count:5d}  ({count/len(merged)*100:5.1f}%)")

    print(f"\n  Final-team distribution:")
    for team, count in merged['label_final'].value_counts().sort_values(ascending=False).items():
        print(f"    {team:30s}  {count:5d}  ({count/len(merged)*100:5.1f}%)")

    # ── Clean text ───────────────────────────────────────────────────────
    print("\n[3/7] Cleaning text...")
    merged['request_text_clean'] = merged['request_text'].apply(clean_text)

    # ── Build features ───────────────────────────────────────────────────
    print("\n[4/7] Building features...")
    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(merged['request_text_clean'])
    structured_matrix = build_structured_features(merged)
    X = sp.hstack([tfidf_matrix, structured_matrix])
    print(f"  Feature shape: {X.shape}")

    y_bot = merged['label_bot'].values
    y_final = merged['label_final'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Cross-validate on bot labels (primary target) ────────────────────
    print("\n[5/7] Cross-validating models on BOT LABELS...")

    models = {
        "LogisticRegression C=1": LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        "LogisticRegression C=5": LogisticRegression(C=5.0, max_iter=2000, random_state=42),
        "LogisticRegression C=10": LogisticRegression(C=10.0, max_iter=2000, random_state=42),
        "LinearSVC C=1": CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000, random_state=42), cv=3),
    }

    best_name = None
    best_score = 0
    best_model = None

    for name, model in models.items():
        scores = cross_val_score(model, X, y_bot, cv=skf, scoring='accuracy', n_jobs=-1)
        print(f"  {name:30s}  accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name
            best_model = model

    print(f"\n  ✓ Best: {best_name} (accuracy = {best_score:.4f})")

    # ── Also check final_team models ─────────────────────────────────────
    print("\n[6/7] Cross-validating on FINAL TEAM for comparison...")

    for name, model in {
        "LR C=1 (final)": LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        "SVC C=1 (final)": CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000, random_state=42), cv=3),
    }.items():
        scores = cross_val_score(model, X, y_final, cv=skf, scoring='accuracy', n_jobs=-1)
        print(f"  {name:30s}  accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

    # Cross-check: how do bot-label predictions fare against final_team?
    cv_preds_bot = cross_val_predict(best_model, X, y_bot, cv=skf)
    print(f"\n  Bot-label model predictions vs bot labels:  {accuracy_score(y_bot, cv_preds_bot):.4f}")
    print(f"  Bot-label model predictions vs final team:   {accuracy_score(y_final, cv_preds_bot):.4f}")
    print(f"  Bot's own labels vs final team:              {accuracy_score(y_final, y_bot):.4f}")

    # ── Train final model ────────────────────────────────────────────────
    print("\n[7/7] Training final model on all data...")
    best_model.fit(X, y_bot)

    train_preds = best_model.predict(X)
    print(f"  Training accuracy: {accuracy_score(y_bot, train_preds):.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_bot, train_preds, digits=4))

    # ── Save ─────────────────────────────────────────────────────────────
    preprocessor = {
        'tfidf': tfidf,
        'valid_channels': VALID_CHANNELS,
        'valid_products': VALID_PRODUCTS,
        'valid_warranty': VALID_WARRANTY,
        'keyword_features': KEYWORD_FEATURES,
        'team_rename': TEAM_RENAME,
        'canonical_teams': CANONICAL_TEAMS,
    }

    joblib.dump(best_model, os.path.join(MODEL_DIR, "router_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

    # Also save the final_team model for comparison
    final_model = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=3000, random_state=42), cv=3
    )
    final_model.fit(X, y_final)
    joblib.dump(final_model, os.path.join(MODEL_DIR, "router_model_final_team.pkl"))

    print(f"\n  Primary model (bot-labels) saved to: models/router_model.pkl")
    print(f"  Final-team model saved to: models/router_model_final_team.pkl")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Primary model: {best_name}")
    print(f"  CV accuracy (vs bot labels): {best_score:.4f}")
    print(f"  Bot misroute rate: {misroute_rate:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
