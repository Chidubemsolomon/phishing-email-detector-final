"""Loading and inference helpers for the saved phishing model."""
from __future__ import annotations

from pathlib import Path

import joblib
from scipy.sparse import hstack

from .features import build_numeric_features, combined_text, feature_reasons


def load_bundle(model_path: str | Path):
    return joblib.load(model_path)


def predict_email(bundle, sender: str, subject: str, body: str, expected_domain: str = "") -> dict:
    text = combined_text(subject, body)
    text_matrix = bundle["vectorizer"].transform([text])
    numeric = build_numeric_features(sender, subject, body)[bundle["numeric_columns"]]
    numeric_matrix = bundle["scaler"].transform(numeric.to_numpy(dtype=float))
    X = hstack([text_matrix, numeric_matrix], format="csr")
    probability = float(bundle["model"].predict_proba(X)[0, 1])
    label = "PHISHING / SUSPICIOUS" if probability >= 0.50 else "LEGITIMATE"
    risk = round(probability * 100, 1)
    return {
        "label": label,
        "probability": probability,
        "risk_score": risk,
        "reasons": feature_reasons(sender, subject, body, expected_domain),
    }
