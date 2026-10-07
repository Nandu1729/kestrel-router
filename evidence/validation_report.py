"""
Kestrel Home — Validation Report Generator
Produces thorough evidence of model performance.
Uses the PRIMARY model (trained on bot labels) and compares against both
bot labels and final outcomes.
"""

import os, re, sys, warnings
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

TEAM_RENAME = {"Installations": "Installs & Demo", "Consumables": "Filters & Consumables"}
VALID_CHANNELS = ["ivr", "chat", "whatsapp", "email"]
VALID_PRODUCTS = ["Air Fryer", "Ceiling Fan", "Induction Cooktop", "Mixer Grinder", "Robot Vacuum", "Room Heater", "Water Purifier"]
VALID_WARRANTY = ["in_warranty", "out_of_warranty", "shield"]
KEYWORD_FEATURES = [
    'install', 'demo', 'wall mount', 'mounting', 'delivered but not installed',
    'filter', 'spare', 'blade', 'jar', 'membrane', 'candle', 'brush', 'amc', 'consumable',
    'invoice', 'refund', 'emi', 'coupon', 'gst', 'double charge', 'payment', 'receipt',
    'return', 'cancel', 'exchange', 'missing parts', 'damaged', 'wrong product', 'pickup',
    'warranty', 'claim', 'shield', 'coverage', 'registration', 'warranty card',
    'error code', 'noise', 'leak', 'not working', 'broken', 'tripping', 'stopped working', 'blank', 'fault', 'smell', 'smoke',
    'how to', 'best settings', 'recipe', 'power consumption', 'clean', 'temperature', 'usage',
    'call back', 'call me', 'please call',
]


def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', text).strip().lower()


def normalize_team(name):
    return TEAM_RENAME.get(name, name)


def build_structured(df):
    n = len(df)
    ch = np.zeros((n, len(VALID_CHANNELS)))
    for i, c in enumerate(df['channel'].values):
        if c in VALID_CHANNELS: ch[i, VALID_CHANNELS.index(c)] = 1
    pr = np.zeros((n, len(VALID_PRODUCTS)))
    for i, p in enumerate(df['product_family'].values):
        if p in VALID_PRODUCTS: pr[i, VALID_PRODUCTS.index(p)] = 1
    wa = np.zeros((n, len(VALID_WARRANTY)))
    for i, w in enumerate(df['warranty_status'].values):
        if w in VALID_WARRANTY: wa[i, VALID_WARRANTY.index(w)] = 1
    kw = np.zeros((n, len(KEYWORD_FEATURES)))
    for i, text in enumerate(df['request_text_clean'].values):
        for j, k in enumerate(KEYWORD_FEATURES):
            if k in text: kw[i, j] = 1
    return sp.csr_matrix(np.hstack([ch, pr, wa, kw]))


def main():
    output_lines = []
    def p(text=""):
        print(text)
        output_lines.append(text)

    p("=" * 78)
    p("KESTREL HOME — VALIDATION REPORT")
    p(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M IST')}")
    p("=" * 78)

    # ── Load data ────────────────────────────────────────────────────────
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    res = pd.read_csv(os.path.join(DATA_DIR, "resolution_log.csv"))
    merged = train_df.merge(res[['request_id', 'final_team', 'transfers']], on='request_id', how='left')
    merged['label_bot'] = merged['team_label'].apply(normalize_team)
    merged['label_final'] = merged['final_team'].apply(normalize_team)
    merged['request_text_clean'] = merged['request_text'].apply(clean_text)

    # ── Build features (same pipeline as train_model.py) ─────────────────
    preproc = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    tfidf = preproc['tfidf']
    tfidf_mat = tfidf.transform(merged['request_text_clean'])
    struct = build_structured(merged)
    X = sp.hstack([tfidf_mat, struct])

    y_bot = merged['label_bot'].values
    y_final = merged['label_final'].values
    teams = sorted(set(y_bot))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Recreate model for cross-validation (same hyperparams) ───────────
    model = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=3000, random_state=42), cv=3
    )
    cv_preds = cross_val_predict(model, X, y_bot, cv=skf)

    # ── §1 Cross-Validation vs Bot Labels ────────────────────────────────
    p("\n" + "─" * 78)
    p("§1  CROSS-VALIDATION vs BOT LABELS (5-fold stratified)")
    p("    This is the primary metric: how well we reproduce the bot's routing.")
    p("─" * 78)

    cv_acc = accuracy_score(y_bot, cv_preds)
    cv_f1 = f1_score(y_bot, cv_preds, average='weighted')

    p(f"\n  Overall accuracy:    {cv_acc:.4f}  ({cv_acc*100:.1f}%)")
    p(f"  Weighted F1:         {cv_f1:.4f}")
    p(f"  Target:              ≥ 90% (Ritu's bar)")
    p(f"  Result:              {'✓ PASS' if cv_acc >= 0.90 else '✗ FAIL'}")

    p(f"\n  Per-fold accuracy:")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_bot)):
        fold_acc = accuracy_score(y_bot[val_idx], cv_preds[val_idx])
        p(f"    Fold {fold_idx+1}: {fold_acc:.4f}")

    # ── §2 Classification Report vs Bot Labels ──────────────────────────
    p("\n" + "─" * 78)
    p("§2  CLASSIFICATION REPORT vs BOT LABELS")
    p("─" * 78)
    p(classification_report(y_bot, cv_preds, digits=4))

    # ── §3 Confusion Matrix vs Bot Labels ────────────────────────────────
    p("\n" + "─" * 78)
    p("§3  CONFUSION MATRIX vs BOT LABELS")
    p("    Rows = actual bot label, Columns = our prediction")
    p("─" * 78)
    cm = confusion_matrix(y_bot, cv_preds, labels=teams)
    header = f"{'':>25s}" + "".join(f"{t[:10]:>12s}" for t in teams)
    p(f"\n{header}")
    p("  " + "-" * (25 + 12 * len(teams)))
    for i, team in enumerate(teams):
        row = f"{team:>25s}" + "".join(f"{cm[i,j]:>12d}" for j in range(len(teams)))
        p(row)

    # ── §4 Cross-evaluation: our model vs final outcomes ─────────────────
    p("\n" + "─" * 78)
    p("§4  OUR MODEL vs ACTUAL OUTCOMES (final_team)")
    p("    How would our model perform against where requests actually ended up?")
    p("─" * 78)

    our_vs_final = accuracy_score(y_final, cv_preds)
    bot_vs_final = accuracy_score(y_final, y_bot)
    p(f"\n  Bot accuracy vs final outcomes:   {bot_vs_final:.4f}  ({bot_vs_final*100:.1f}%)")
    p(f"  Our accuracy vs final outcomes:   {our_vs_final:.4f}  ({our_vs_final*100:.1f}%)")
    p(f"  Improvement:                      {(our_vs_final - bot_vs_final)*100:+.1f} pp")

    # Where we differ from bot labels — are we right or wrong?
    differ_mask = cv_preds != y_bot
    n_differ = differ_mask.sum()
    p(f"\n  Cases where our model disagrees with bot: {n_differ}")
    # Of those, how many is our model correct (matches final_team)?
    our_correct_when_differ = (cv_preds[differ_mask] == y_final[differ_mask]).sum()
    bot_correct_when_differ = (y_bot[differ_mask] == y_final[differ_mask]).sum()
    p(f"    Our model correct (matches final):  {our_correct_when_differ} ({our_correct_when_differ/n_differ*100:.1f}%)")
    p(f"    Bot correct (matches final):        {bot_correct_when_differ} ({bot_correct_when_differ/n_differ*100:.1f}%)")
    p(f"    Neither correct:                    {n_differ - our_correct_when_differ - bot_correct_when_differ}")

    # ── §5 Transfer & Cost Analysis ──────────────────────────────────────
    p("\n" + "─" * 78)
    p("§5  TRANSFER & COST ANALYSIS")
    p("─" * 78)

    bot_misroutes = (y_bot != y_final).sum()
    our_misroutes = (cv_preds != y_final).sum()
    monthly_vol = 700
    transfer_cost = 305 + 260  # Rs per misroute (transfer + extra contact)

    p(f"\n  Bot misroutes:  {bot_misroutes}/{len(merged)} ({bot_misroutes/len(merged)*100:.1f}%)")
    p(f"  Our misroutes:  {our_misroutes}/{len(merged)} ({our_misroutes/len(merged)*100:.1f}%)")
    p(f"  Transfer reduction: {bot_misroutes - our_misroutes} fewer transfers per {len(merged)} requests")

    monthly_bot_misroutes = monthly_vol * (bot_misroutes / len(merged))
    monthly_our_misroutes = monthly_vol * (our_misroutes / len(merged))
    annual_bot_cost = monthly_bot_misroutes * 12 * transfer_cost
    annual_our_cost = monthly_our_misroutes * 12 * transfer_cost
    bot_licence = 320000

    p(f"\n  Monthly volume: ~{monthly_vol} requests")
    p(f"  Cost per misroute: Rs {transfer_cost} (Rs 305 transfer + Rs 260 extra contact)")
    p(f"")
    p(f"  Current bot:")
    p(f"    Monthly misroutes:  ~{monthly_bot_misroutes:.0f}")
    p(f"    Annual misroute cost: Rs {annual_bot_cost:,.0f}")
    p(f"    Annual licence:       Rs {bot_licence:,}")
    p(f"    Total annual cost:    Rs {annual_bot_cost + bot_licence:,.0f}")
    p(f"")
    p(f"  Our model:")
    p(f"    Monthly misroutes:  ~{monthly_our_misroutes:.0f}")
    p(f"    Annual misroute cost: Rs {annual_our_cost:,.0f}")
    p(f"    Annual licence:       Rs 0")
    p(f"    Total annual cost:    Rs {annual_our_cost:,.0f}")
    p(f"")
    p(f"  NET ANNUAL SAVINGS:   Rs {annual_bot_cost + bot_licence - annual_our_cost:,.0f}")

    # ── §6 Error Analysis ────────────────────────────────────────────────
    p("\n" + "─" * 78)
    p("§6  ERROR ANALYSIS (mismatches vs bot labels)")
    p("─" * 78)

    errors_mask = cv_preds != y_bot
    errors = merged[errors_mask].copy()
    errors['predicted'] = cv_preds[errors_mask]
    errors['actual_bot'] = y_bot[errors_mask]
    errors['actual_final'] = y_final[errors_mask]

    p(f"\n  Total mismatches vs bot: {errors_mask.sum()} ({errors_mask.mean()*100:.1f}%)")

    p("\n  Most common mismatches (bot_label → our_prediction):")
    pairs = errors.groupby(['actual_bot', 'predicted']).size().sort_values(ascending=False).head(10)
    for (actual, pred), count in pairs.items():
        p(f"    {actual:>25s} → {pred:<25s}  {count:4d}")

    p("\n  Error rate by channel:")
    for ch in VALID_CHANNELS:
        mask = merged['channel'].values == ch
        errs = (cv_preds[mask] != y_bot[mask]).sum()
        total = mask.sum()
        if total > 0:
            p(f"    {ch:>12s}: {errs}/{total} ({errs/total*100:.1f}%)")

    p("\n  Error rate by source:")
    for src in ['crm', 'legacy_zoho']:
        mask = merged['source'].values == src
        errs = (cv_preds[mask] != y_bot[mask]).sum()
        total = mask.sum()
        if total > 0:
            p(f"    {src:>12s}: {errs}/{total} ({errs/total*100:.1f}%)")

    # ── §7 Hard Cases ────────────────────────────────────────────────────
    p("\n" + "─" * 78)
    p("§7  SAMPLE HARD CASES")
    p("─" * 78)

    # Cases where both bot AND our model are wrong (vs final_team)
    both_wrong = (cv_preds != y_final) & (y_bot != y_final)
    both_wrong_df = merged[both_wrong].head(5)
    p("\n  Cases where BOTH bot and our model are wrong vs final outcome:")
    for _, row in both_wrong_df.iterrows():
        idx = merged.index.get_loc(row.name)
        p(f"    {row['request_id']}: \"{row['request_text'][:80]}...\"")
        p(f"      Bot: {y_bot[idx]}, Ours: {cv_preds[idx]}, Final: {y_final[idx]}")

    # Ambiguous requests (low confidence)
    model_trained = joblib.load(os.path.join(MODEL_DIR, "router_model.pkl"))
    probas = model_trained.predict_proba(X)
    max_conf = probas.max(axis=1)
    low_conf_idx = np.argsort(max_conf)[:5]
    p("\n  Lowest-confidence predictions:")
    for idx in low_conf_idx:
        row = merged.iloc[idx]
        p(f"    {row['request_id']}: conf={max_conf[idx]:.3f}")
        p(f"      \"{row['request_text'][:80]}...\"")
        p(f"      Predicted: {model_trained.predict(X[idx])[0]}, Bot: {y_bot[idx]}, Final: {y_final[idx]}")

    # ── Save ─────────────────────────────────────────────────────────────
    report_path = os.path.join(EVIDENCE_DIR, "validation_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(output_lines))

    p(f"\n\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
