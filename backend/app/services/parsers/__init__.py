from pathlib import Path
from typing import BinaryIO

from app.models.document_object import DocumentObject
from app.services.parsers.docx_parser import parse_docx
from app.services.parsers.pdf_parser import parse_pdf


def parse_document(file_path: str | Path, filename: str) -> DocumentObject:
    path = Path(file_path)
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".docx", ".doc"}:
        return parse_docx(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def parse_upload(filename: str, content: BinaryIO) -> DocumentObject:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(content)
    if suffix in {".docx", ".doc"}:
        return parse_docx(content)
    raise ValueError(f"Unsupported file type: {suffix}")
