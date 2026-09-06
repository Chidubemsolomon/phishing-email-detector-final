"""Train the phishing-email detector from a local public dataset file.

Expected input: data/phishing_raw.jsonl in the format published by
https://huggingface.co/datasets/darkknight25/phishing_benign_email_dataset

This script NEVER downloads data. The dataset must be supplied locally.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features import build_numeric_features, combined_text

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "phishing_raw.jsonl"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "phishing_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

LABELS = {"phishing": 1, "benign": 0, "legitimate": 0, "safe": 0}
NUMERIC_COLUMNS = [
    "sender_valid", "sender_domain_length", "sender_has_hyphen", "sender_has_digit",
    "sender_subdomain_count", "sender_free_provider", "url_count", "url_https_ratio",
    "url_ip_count", "url_at_count", "url_suspicious_structure_count", "max_url_host_length",
    "urgent_term_count", "sensitive_term_count", "body_length", "subject_length",
    "exclamation_count", "digit_ratio"
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Upload the real public JSONL dataset as data/phishing_raw.jsonl."
        )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    required = {"subject", "body", "spoofed_sender", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
    df = df[["subject", "body", "spoofed_sender", "label"]].copy()
    for col in ["subject", "body", "spoofed_sender"]:
        df[col] = df[col].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABELS)].copy()
    df = df.drop_duplicates(subset=["subject", "body", "spoofed_sender", "label"]).reset_index(drop=True)
    if df["label"].nunique() < 2:
        raise ValueError("The dataset must contain both phishing and benign examples.")
    df["target"] = df["label"].map(LABELS)
    df["text"] = [combined_text(s, b) for s, b in zip(df.subject, df.body)]
    return df


def make_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    frames = [
        build_numeric_features(sender, subject, body)
        for sender, subject, body in zip(df.spoofed_sender, df.subject, df.body)
    ]
    return pd.concat(frames, ignore_index=True)[NUMERIC_COLUMNS].to_numpy(dtype=float)


def main() -> None:
    df = load_dataset(DATA_PATH)
    X_train_text, X_test_text, X_train_df, X_test_df, y_train, y_test = train_test_split(
        df["text"],
        df,
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_features=6000,
        sublinear_tf=True,
    )
    X_train_text_mat = vectorizer.fit_transform(X_train_text)
    X_test_text_mat = vectorizer.transform(X_test_text)

    scaler = StandardScaler()
    train_num = scaler.fit_transform(make_numeric_matrix(X_train_df))
    test_num = scaler.transform(make_numeric_matrix(X_test_df))

    X_train = hstack([X_train_text_mat, train_num], format="csr")
    X_test = hstack([X_test_text_mat, test_num], format="csr")

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.50).astype(int)

    metrics = {
        "dataset_size": int(len(df)),
        "class_counts": {"benign": int((df.target == 0).sum()), "phishing": int((df.target == 1).sum())},
        "train_size": int(len(X_train_df)),
        "test_size": int(len(X_test_df)),
        "random_state": 42,
        "test_size_fraction": 0.20,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, target_names=["benign", "phishing"], output_dict=True, zero_division=0
        ),
        "model": "TF-IDF + numeric security features + Logistic Regression",
        "tfidf_features": int(len(vectorizer.vocabulary_)),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "scaler": scaler,
            "model": model,
            "numeric_columns": NUMERIC_COLUMNS,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({k: metrics[k] for k in ["dataset_size", "train_size", "test_size", "accuracy", "precision", "recall", "f1", "roc_auc"]}, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
