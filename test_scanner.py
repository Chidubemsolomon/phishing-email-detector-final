from src.features import build_numeric_features, extract_urls, feature_reasons


def test_extract_urls():
    urls = extract_urls("Visit https://example.com and www.example.org now")
    assert len(urls) == 2


def test_numeric_features_shape():
    df = build_numeric_features(
        "student@example.com",
        "Hello",
        "Please read the update at https://example.com",
    )
    assert len(df) == 1
    assert df.loc[0, "url_count"] == 1
    assert "sender_valid" in df.columns


def test_reasons_detect_url_and_urgency():
    reasons = feature_reasons(
        "security@example.com",
        "URGENT: verify now",
        "Click https://123.123.123.123/login immediately",
    )
    joined = " ".join(reasons).lower()
    assert "url" in joined or "urgent" in joined
