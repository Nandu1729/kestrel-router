"""
Kestrel Home — Prediction Generator
Loads the trained model and generates predictions for test_unlabelled.csv.
Outputs predictions.csv in the sample_submission.csv format.
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def clean_text(text):
    """Clean garbled Unicode and normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def main():
    print("=" * 70)
    print("KESTREL HOME — PREDICTION GENERATOR")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading model...")
    model = joblib.load(os.path.join(MODEL_DIR, "router_model.pkl"))
    preproc = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    print("  Model loaded successfully.")

    tfidf = preproc['tfidf']
    VALID_CHANNELS = preproc['valid_channels']
    VALID_PRODUCTS = preproc['valid_products']
    VALID_WARRANTY = preproc['valid_warranty']
    KEYWORD_FEATURES = preproc['keyword_features']

    # Load test data
    print("\n[2/3] Loading and featurizing test data...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_unlabelled.csv"))
    print(f"  Test rows: {len(test_df)}")

    # Clean text
    test_df['request_text_clean'] = test_df['request_text'].apply(clean_text)

    # TF-IDF
    tfidf_matrix = tfidf.transform(test_df['request_text_clean'])

    # Structured features
    n = len(test_df)
    channel_mat = np.zeros((n, len(VALID_CHANNELS)))
    for i, ch in enumerate(test_df['channel'].values):
        if ch in VALID_CHANNELS:
            channel_mat[i, VALID_CHANNELS.index(ch)] = 1

    product_mat = np.zeros((n, len(VALID_PRODUCTS)))
    for i, p in enumerate(test_df['product_family'].values):
        if p in VALID_PRODUCTS:
            product_mat[i, VALID_PRODUCTS.index(p)] = 1

    warranty_mat = np.zeros((n, len(VALID_WARRANTY)))
    for i, w in enumerate(test_df['warranty_status'].values):
        if w in VALID_WARRANTY:
            warranty_mat[i, VALID_WARRANTY.index(w)] = 1

    texts = test_df['request_text_clean'].values
    keyword_mat = np.zeros((n, len(KEYWORD_FEATURES)))
    for i, text in enumerate(texts):
        for j, kw in enumerate(KEYWORD_FEATURES):
            if kw in text:
                keyword_mat[i, j] = 1

    structured = np.hstack([channel_mat, product_mat, warranty_mat, keyword_mat])
    X_test = sp.hstack([tfidf_matrix, sp.csr_matrix(structured)])
    print(f"  Feature matrix: {X_test.shape}")

    # Predict
    print("\n[3/3] Generating predictions...")
    predictions = model.predict(X_test)

    # Verify all predictions are valid team names
    canonical = set(preproc['canonical_teams'])
    pred_set = set(predictions)
    assert pred_set.issubset(canonical), \
        f"Invalid team names in predictions: {pred_set - canonical}"

    # Distribution
    pred_series = pd.Series(predictions)
    print(f"\n  Prediction distribution:")
    for team, count in pred_series.value_counts().sort_values(ascending=False).items():
        pct = count / len(pred_series) * 100
        print(f"    {team:30s}  {count:5d}  ({pct:5.1f}%)")

    # Save
    output_path = os.path.join(BASE_DIR, "predictions.csv")
    submission = pd.DataFrame({
        'request_id': test_df['request_id'],
        'team': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    print(f"  Rows: {len(submission)}")

    # Confidence stats
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)
        max_conf = probas.max(axis=1)
        print(f"\n  Confidence stats:")
        print(f"    Mean: {max_conf.mean():.3f}")
        print(f"    Median: {np.median(max_conf):.3f}")
        print(f"    Min: {max_conf.min():.3f}")
        print(f"    < 50%: {(max_conf < 0.5).sum()} requests ({(max_conf < 0.5).mean():.1%})")
        print(f"    < 70%: {(max_conf < 0.7).sum()} requests ({(max_conf < 0.7).mean():.1%})")

    print("\n" + "=" * 70)
    print("PREDICTIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
