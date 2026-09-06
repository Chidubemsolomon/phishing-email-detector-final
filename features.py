"""Feature extraction for phishing-email detection.

This module never opens or visits URLs. URLs are treated as plain text.
"""
from __future__ import annotations

import re
from email.utils import parseaddr
from urllib.parse import urlparse

import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
IP_HOST_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "proton.me", "protonmail.com", "aol.com", "live.com", "msn.com"
}
URGENT_WORDS = {
    "urgent", "immediately", "verify", "suspended", "suspend", "locked",
    "disable", "disabled", "expire", "expired", "action required", "reset",
    "confirm now", "act now", "final notice", "security alert"
}
SENSITIVE_WORDS = {
    "password", "passcode", "bank", "credit card", "debit card", "otp",
    "verification code", "login", "credential", "ssn", "social security"
}


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text or "")


def _domain_from_email(sender: str) -> str:
    _name, address = parseaddr(sender or "")
    if "@" not in address:
        return ""
    return address.rsplit("@", 1)[-1].strip().lower()


def _valid_email(sender: str) -> bool:
    _name, address = parseaddr(sender or "")
    return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", address))


def _url_host(url: str) -> str:
    value = url if re.match(r"^https?://", url, re.I) else f"http://{url}"
    try:
        return (urlparse(value).hostname or "").lower()
    except ValueError:
        return ""


def suspicious_url_structure(url: str) -> bool:
    host = _url_host(url)
    lowered = url.lower()
    return any([
        "@" in lowered,
        bool(IP_HOST_RE.fullmatch(host)),
        host.count("-") >= 3,
        host.count(".") >= 4,
        len(host) > 35,
    ])


def build_numeric_features(sender: str, subject: str, body: str) -> pd.DataFrame:
    text = f"{subject or ''} {body or ''}"
    urls = extract_urls(text)
    domain = _domain_from_email(sender)

    url_hosts = [_url_host(u) for u in urls]
    https_count = sum(1 for u in urls if u.lower().startswith("https://"))
    ip_url_count = sum(1 for h in url_hosts if IP_HOST_RE.fullmatch(h or ""))
    at_url_count = sum(1 for u in urls if "@" in u)
    suspicious_url_count = sum(1 for u in urls if suspicious_url_structure(u))

    values = {
        "sender_valid": int(_valid_email(sender)),
        "sender_domain_length": len(domain),
        "sender_has_hyphen": int("-" in domain),
        "sender_has_digit": int(any(ch.isdigit() for ch in domain)),
        "sender_subdomain_count": max(domain.count(".") - 1, 0) if domain else 0,
        "sender_free_provider": int(domain in FREE_EMAIL_PROVIDERS),
        "url_count": len(urls),
        "url_https_ratio": (https_count / len(urls)) if urls else 0.0,
        "url_ip_count": ip_url_count,
        "url_at_count": at_url_count,
        "url_suspicious_structure_count": suspicious_url_count,
        "max_url_host_length": max((len(h) for h in url_hosts), default=0),
        "urgent_term_count": sum(1 for term in URGENT_WORDS if term in text.lower()),
        "sensitive_term_count": sum(1 for term in SENSITIVE_WORDS if term in text.lower()),
        "body_length": len(body or ""),
        "subject_length": len(subject or ""),
        "exclamation_count": text.count("!"),
        "digit_ratio": (sum(ch.isdigit() for ch in text) / max(len(text), 1)),
    }
    return pd.DataFrame([values])


def combined_text(subject: str, body: str) -> str:
    return f"subject: {subject or ''}\nbody: {body or ''}".strip()


def feature_reasons(sender: str, subject: str, body: str, expected_domain: str = "") -> list[str]:
    text = f"{subject or ''} {body or ''}"
    urls = extract_urls(text)
    domain = _domain_from_email(sender)
    reasons: list[str] = []

    if not _valid_email(sender):
        reasons.append("The sender address does not appear to have a normal email format.")
    if domain in FREE_EMAIL_PROVIDERS:
        reasons.append("The sender uses a common free-email provider.")
    if urls:
        reasons.append(f"The email contains {len(urls)} URL(s).")
    if any(IP_HOST_RE.fullmatch(_url_host(u) or "") for u in urls):
        reasons.append("A URL uses an IP address instead of a normal domain name.")
    if any("@" in u for u in urls):
        reasons.append("A URL contains the @ symbol, which can be used to hide the real destination.")
    if any(suspicious_url_structure(u) for u in urls):
        reasons.append("One or more URLs have suspicious structural characteristics.")
    lowered = text.lower()
    if any(term in lowered for term in URGENT_WORDS):
        reasons.append("The message contains urgent or account-action language.")
    if any(term in lowered for term in SENSITIVE_WORDS):
        reasons.append("The message asks about login, password, financial, or other sensitive information.")
    if expected_domain.strip() and domain and domain != expected_domain.strip().lower():
        reasons.append("The sender domain does not match the expected organization domain provided.")

    if not reasons:
        reasons.append("No strong rule-based warning indicators were found.")
    return reasons[:6]
