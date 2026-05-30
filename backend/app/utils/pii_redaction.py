"""
Strip obvious PII before text goes to DB, FAISS, or Gemini.

We run this at parse time so sensitive data doesn't sit in indexes longer than needed.
Not a substitute for proper compliance — but shows we thought about privacy in design.
"""

import re

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")
PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")


def redact_pii(text: str) -> str:
    text = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
    text = PHONE_PATTERN.sub("[PHONE_REDACTED]", text)
    text = PAN_PATTERN.sub("[PAN_REDACTED]", text)
    text = AADHAAR_PATTERN.sub("[AADHAAR_REDACTED]", text)
    return text
