LANGUAGE_MAP = {
    "en": "English",
    "english": "English",
    "hi": "Hindi",
    "hindi": "Hindi",
    "hinglish": "Hinglish (mix of Hindi and English)",
}


def normalize_language(code: str) -> str:
    return LANGUAGE_MAP.get(code.lower().strip(), "English")


def cross_lingual_instruction(question_lang: str, doc_lang: str) -> str:
    if question_lang == doc_lang:
        return ""
    return (
        f"The user question is in {question_lang} and the document context may be in {doc_lang}. "
        "Answer in the user's question language. Translate context faithfully; do not invent clauses."
    )
