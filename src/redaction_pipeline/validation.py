from __future__ import annotations

from pathlib import Path

import pymupdf
from pypdf import PdfReader

from .models import ValidationResult


class ValidationError(ValueError):
    pass


def validate_pdf(path: str | Path, max_size_bytes: int = 50 * 1024 * 1024) -> ValidationResult:
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"PDF does not exist: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValidationError(f"Expected a .pdf file: {source}")
    size = source.stat().st_size
    if size == 0 or size > max_size_bytes:
        raise ValidationError(f"PDF size {size} is outside the allowed range 1..{max_size_bytes}")

    try:
        reader = PdfReader(str(source), strict=True)
        encrypted = reader.is_encrypted
        if encrypted and not reader.decrypt(""):
            raise ValidationError("Password-protected PDFs are not accepted")
        expected_pages = len(reader.pages)
        with pymupdf.open(source) as pdf:
            if pdf.needs_pass and not pdf.authenticate(""):
                raise ValidationError("Password-protected PDFs are not accepted")
            if pdf.page_count < 1:
                raise ValidationError("PDF has no pages")
            if pdf.page_count != expected_pages:
                raise ValidationError("PDF parsers disagree on page count")
            has_metadata = any((pdf.metadata or {}).values())
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Unreadable or corrupt PDF: {exc}") from exc

    return ValidationResult(str(source), size, expected_pages, encrypted, has_metadata)
