from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from model_utils import load_bundle, predict_email

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "phishing_model.joblib"
METRICS_PATH = ROOT / "metrics.json"
DATA_CANDIDATES = [
    ROOT / "phishing_raw.jsonl",
    ROOT / "data" / "phishing_raw.jsonl",
]

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ AI-Powered Phishing Email Detection")
st.caption(
    "A machine-learning prototype for analysing email content, sender information, and URL structure."
)


def ensure_model() -> bool:
    """Train once automatically if the saved model is missing."""
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        return True

    dataset_path = next((p for p in DATA_CANDIDATES if p.exists()), None)

    if dataset_path is None:
        st.error(
            "The trained model is missing and the project dataset was not found."
        )
        st.info(
            "Make sure phishing_raw.jsonl is uploaded to the GitHub repository "
            "in the same folder as app.py."
        )
        return False

    try:
        # train.py accepts the dataset in the project root. If it is inside
        # data/, copy it to the root temporarily so the existing training code
        # can be reused without creating a second training implementation.
        root_dataset = ROOT / "phishing_raw.jsonl"
        copied = False

        if dataset_path != root_dataset:
            root_dataset.write_text(
                dataset_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            copied = True

        from train import main as train_model

        with st.spinner("Preparing the trained model for first use..."):
            train_model()

        if copied and root_dataset.exists():
            root_dataset.unlink()

        return MODEL_PATH.exists() and METRICS_PATH.exists()

    except Exception as exc:
        st.error("The model could not be prepared automatically.")
        st.exception(exc)
        return False


if not ensure_model():
    st.stop()


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


bundle = get_bundle()

scanner, performance, about = st.tabs(
    ["Email Scanner", "Model Performance", "About"]
)

with scanner:
    sender = st.text_input(
        "Sender email",
        placeholder="example@company.com",
    )
    subject = st.text_input(
        "Subject",
        placeholder="Your account notification",
    )
    body = st.text_area(
        "Email body",
        height=220,
        placeholder="Paste the email message here...",
    )
    expected_domain = st.text_input(
        "Expected organization domain (optional)",
        placeholder="company.com",
    )

    col1, col2 = st.columns(2)

    with col1:
        scan = st.button("🔎 Scan Email", use_container_width=True)

    with col2:
        clear = st.button("Clear", use_container_width=True)
        if clear:
            st.rerun()

    if scan:
        if not sender.strip() and not subject.strip() and not body.strip():
            st.error("Enter at least the email sender, subject, or body.")
        else:
            result = predict_email(
                bundle,
                sender,
                subject,
                body,
                expected_domain,
            )

            st.divider()

            if result["label"].startswith("PHISHING"):
                st.error(f"⚠️ {result['label']}")
            else:
                st.success(f"✅ {result['label']}")

            st.metric("Risk score", f"{result['risk_score']:.1f}%")

            st.write("**Warning signs / observations**")
            for reason in result["reasons"]:
                st.write(f"• {reason}")

            st.caption(
                "This result is a model-assisted warning, not a guarantee "
                "that an email is safe or malicious."
            )

with performance:
    metrics = json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )

    a, p, r, f = st.columns(4)
    a.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    p.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    r.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    f.metric("F1-score", f"{metrics['f1'] * 100:.1f}%")

    st.write(f"**Dataset used:** {metrics['dataset_size']} emails")
    st.write(
        f"**Training / Test:** {metrics['train_size']} / "
        f"{metrics['test_size']}"
    )

    st.write("**Confusion matrix**")
    st.table(
        {
            "Actual benign": metrics["confusion_matrix"][0],
            "Actual phishing": metrics["confusion_matrix"][1],
        }
    )

    st.caption("Metrics are from the held-out test set used by train.py.")

with about:
    st.subheader("What this system does")
    st.write(
        "The system uses TF-IDF to turn email text into numerical features "
        "and Logistic Regression to estimate whether the message is phishing "
        "or legitimate. It also checks simple sender and URL characteristics."
    )

    st.subheader("Safety")
    st.write(
        "URLs are inspected only as text. The application never opens or "
        "visits user-supplied links."
    )

    st.subheader("Limitation")
    st.write(
        "The prototype cannot guarantee detection of every phishing email. "
        "Performance depends on the quality and variety of the training data."
    )
