import json
import logging
import re

import google.generativeai as genai

from app.config import get_settings
from app.utils.language import normalize_language

logger = logging.getLogger(__name__)
settings = get_settings()

SUMMARY_PROMPT = """Analyze this legal contract and return JSON only in {language}.

Contract:
{text}

JSON schema:
{{
  "executive_summary": "string",
  "key_obligations": ["string"],
  "important_dates": ["string"],
  "payment_terms": "string",
  "termination_conditions": "string",
  "risks": ["string"],
  "legal_concerns": ["string"]
}}
"""


class ContractSummarizer:
    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

    def summarize(self, text: str, language: str = "en") -> dict:
        lang = normalize_language(language)
        excerpt = text[:25000]
        prompt = SUMMARY_PROMPT.format(language=lang, text=excerpt)
        response = self.model.generate_content(prompt)
        raw = response.text or ""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {
            "executive_summary": raw[:2000],
            "key_obligations": [],
            "important_dates": [],
            "payment_terms": "",
            "termination_conditions": "",
            "risks": [],
            "legal_concerns": [],
        }
