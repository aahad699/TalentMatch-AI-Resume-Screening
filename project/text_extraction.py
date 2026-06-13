"""
Text extraction utilities for uploaded resume files.

TXT works with no extra dependencies. PDF and DOCX are supported when pypdf and
python-docx are installed.
"""

from __future__ import annotations

import io
from typing import BinaryIO


class TextExtractionError(ValueError):
    """Raised when an uploaded file cannot be converted to text."""


def extract_text_from_upload(file_obj: BinaryIO, filename: str) -> str:
    """Extract text from a Streamlit uploaded file or file-like object."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else "txt"
    try:
        file_obj.seek(0)
    except (AttributeError, OSError):
        pass

    try:
        data = file_obj.read()
    except OSError as exc:
        raise TextExtractionError("Could not read the uploaded file. Please try uploading it again.") from exc

    if not data:
        raise TextExtractionError("Uploaded file is empty.")

    if suffix == "txt":
        return _decode_text(data)
    if suffix == "pdf":
        return _extract_pdf(data)
    if suffix == "docx":
        return _extract_docx(data)

    raise TextExtractionError("Unsupported file type. Upload TXT, PDF, or DOCX.")


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise TextExtractionError("Could not decode text file.")


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise TextExtractionError("PDF upload needs pypdf. Run: pip install pypdf") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        raise TextExtractionError(
            "Could not read this PDF. It may be corrupted, incomplete, password-protected, or image-only."
        ) from exc

    if not text:
        raise TextExtractionError("No selectable text found in this PDF.")
    return text


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise TextExtractionError("DOCX upload needs python-docx. Run: pip install python-docx") from exc

    try:
        doc = Document(io.BytesIO(data))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()
    except Exception as exc:
        raise TextExtractionError("Could not read this DOCX file. It may be corrupted or unsupported.") from exc

    if not text:
        raise TextExtractionError("No text found in this DOCX file.")
    return text
