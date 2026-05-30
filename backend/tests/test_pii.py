from app.utils.pii_redaction import redact_pii


def test_redact_email_and_phone():
    text = "Contact john@example.com or call 9876543210"
    out = redact_pii(text)
    assert "[EMAIL_REDACTED]" in out
    assert "[PHONE_REDACTED]" in out
