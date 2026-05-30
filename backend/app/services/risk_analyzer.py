"""
Find risky clauses in a contract — two passes for better coverage.

Pass 1 (framework):
  load a JSON checklist for this doc type (employment.json, nda.json, …)
  and ask Gemini to flag anything matching those categories.

Pass 2 (open-ended):
  catch weird clauses the checklist might miss — financially/legal/operationally risky.

We merge + dedupe because the same clause sometimes appears in both passes.
Recall matters more than precision here — missing a bad clause is worse than over-flagging.
"""

import json
import logging
import re
from pathlib import Path

import google.generativeai as genai

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

FRAMEWORK_DIR = Path(__file__).resolve().parent.parent / "risk_frameworks"

TYPE_TO_FRAMEWORK = {
    "Employment Contract": "employment.json",
    "Lease Agreement": "lease.json",
    "Rental Agreement": "lease.json",  # rental reuses lease-style risks
    "NDA": "nda.json",
    "Vendor Agreement": "vendor.json",
    "Service Agreement": "service.json",
    "General Legal Document": "general.json",
}


class RiskAnalyzer:
    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

    def _load_framework(self, document_type: str) -> dict:
        fname = TYPE_TO_FRAMEWORK.get(document_type, "general.json")
        path = FRAMEWORK_DIR / fname
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return json.loads((FRAMEWORK_DIR / "general.json").read_text(encoding="utf-8"))

    def analyze(self, text: str, document_type: str, language: str = "en") -> list[dict]:
        framework = self._load_framework(document_type)
        framework_risks = self._framework_extraction(text, framework, document_type)
        open_risks = self._open_ended_discovery(text, language)
        combined = framework_risks + open_risks
        return self._dedupe_risks(combined)

    def _framework_extraction(self, text: str, framework: dict, document_type: str) -> list[dict]:
        risk_types = framework.get("risk_categories", [])
        prompt = f"""You are a legal risk analyst. Document type: {document_type}.
Check for these risk categories: {json.dumps(risk_types)}

For each risk found, return JSON array items:
{{"clause": "...", "explanation": "...", "severity": "Low|Medium|High", "risk_type": "...", "page": null, "section": null}}

Contract excerpt:
{text[:15000]}
"""
        return self._parse_risk_response(prompt, framework_source=framework.get("name", "framework"))

    def _open_ended_discovery(self, text: str, language: str) -> list[dict]:
        prompt = f"""Identify clauses that are financially, legally, or operationally risky or unusually restrictive.
Respond in {language}. Return JSON array only:
[{{"clause": "", "explanation": "", "severity": "Low|Medium|High"}}]

Contract:
{text[:15000]}
"""
        return self._parse_risk_response(prompt, framework_source="open_ended")

    def _parse_risk_response(self, prompt: str, framework_source: str) -> list[dict]:
        try:
            response = self.model.generate_content(prompt)
            raw = response.text or ""
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []
            items = json.loads(match.group())
            for item in items:
                item["framework_source"] = framework_source
            return items
        except Exception as exc:
            logger.warning("Risk parsing failed: %s", exc)
            return []

    def _dedupe_risks(self, risks: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for r in risks:
            key = (r.get("clause") or "")[:80]
            if key in seen:
                continue
            seen.add(key)
            sev = r.get("severity", "Medium")
            if sev not in ("Low", "Medium", "High"):
                r["severity"] = "Medium"
            out.append(r)
        return out
