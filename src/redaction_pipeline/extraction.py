from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Protocol

import pymupdf
import pytesseract
from PIL import Image
from pytesseract import Output

from .models import Document, PageText, Word

_CID_RE = re.compile(r"\[CID \d+\]")


class OCRAdapter(Protocol):
    engine_name: str

    def extract_page(self, page: pymupdf.Page, page_number: int) -> PageText: ...


def _assemble_words(page_number: int, items: list[tuple[str, tuple[float, float, float, float], float]], source: str) -> PageText:
    parts: list[str] = []
    words: list[Word] = []
    cursor = 0
    for token, box, confidence in items:
        token = token.strip()
        if not token:
            continue
        if parts:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(token)
        cursor += len(token)
        words.append(Word(page_number, token, start, cursor, box, confidence))
    reliability = sum(w.confidence for w in words) / len(words) if words else 0.0
    return PageText(page_number, "".join(parts), words, source, reliability)


def _text_layer_page(page: pymupdf.Page, page_number: int) -> PageText:
    raw_words = sorted(page.get_text("words"), key=lambda w: (w[5], w[6], w[7]))
    result = _assemble_words(page_number, [(str(w[4]), tuple(map(float, w[:4])), 1.0) for w in raw_words], "text_layer")
    printable = sum(c.isprintable() or c.isspace() for c in result.text) / max(1, len(result.text))
    density = min(1.0, len(result.text.strip()) / 40.0)
    result.reliability = 0.0 if _CID_RE.search(result.text) else min(printable, density)
    return result


class TesseractOCR:
    engine_name = "tesseract"

    def __init__(self, dpi: int = 300, language: str = "eng"):
        self.dpi = dpi
        self.language = language

    def extract_page(self, page: pymupdf.Page, page_number: int) -> PageText:
        scale = self.dpi / 72.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        data = pytesseract.image_to_data(image, lang=self.language, output_type=Output.DICT, config="--psm 6")
        items: list[tuple[str, tuple[float, float, float, float], float]] = []
        for i, token in enumerate(data["text"]):
            token = token.strip()
            try:
                confidence = max(0.0, float(data["conf"][i]) / 100.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if not token or confidence <= 0:
                continue
            left, top = float(data["left"][i]) / scale, float(data["top"][i]) / scale
            width, height = float(data["width"][i]) / scale, float(data["height"][i]) / scale
            items.append((token, (left, top, left + width, top + height), confidence))
        return _assemble_words(page_number, items, self.engine_name)


class AzureDocumentIntelligenceOCR:
    engine_name = "azure_document_intelligence"

    def __init__(self, endpoint: str, credential):
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
        except ImportError as exc:
            raise RuntimeError("Install Azure support with: pip install -e '.[azure]'") from exc
        self.client = DocumentIntelligenceClient(endpoint=endpoint, credential=credential)

    def extract_page(self, page: pymupdf.Page, page_number: int) -> PageText:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72), alpha=False)
        result = self.client.begin_analyze_document("prebuilt-read", body=io.BytesIO(pix.tobytes("png"))).result()
        analyzed = result.pages[0]
        x_scale, y_scale = page.rect.width / analyzed.width, page.rect.height / analyzed.height
        items: list[tuple[str, tuple[float, float, float, float], float]] = []
        for word in analyzed.words or []:
            polygon = word.polygon or []
            xs, ys = [p.x * x_scale for p in polygon], [p.y * y_scale for p in polygon]
            if xs and ys:
                items.append((word.content, (min(xs), min(ys), max(xs), max(ys)), float(word.confidence or 0.0)))
        return _assemble_words(page_number, items, self.engine_name)


class HybridExtractor:
    """Text layer first; Tesseract primary OCR; optional Azure last fallback."""

    def __init__(self, tesseract: TesseractOCR | None = None, azure: OCRAdapter | None = None, threshold: float = 0.85):
        self.tesseract = tesseract or TesseractOCR()
        self.azure = azure
        self.threshold = threshold

    def extract(self, path: str | Path, force_ocr: bool = False, allow_blank: bool = False) -> Document:
        pages: list[PageText] = []
        with pymupdf.open(path) as pdf:
            for index, page in enumerate(pdf):
                selected = _text_layer_page(page, index + 1)
                if force_ocr or selected.reliability < self.threshold:
                    selected = self.tesseract.extract_page(page, index + 1)
                    if selected.reliability < self.threshold and self.azure is not None:
                        azure_page = self.azure.extract_page(page, index + 1)
                        if azure_page.reliability > selected.reliability:
                            selected = azure_page
                if not selected.words and not allow_blank:
                    raise ValueError(f"Page {index + 1} produced no usable text with configured extraction engines")
                pages.append(selected)
        return Document(str(path), pages)


def extract_pdf(path: str | Path, *, force_ocr: bool = False, azure: OCRAdapter | None = None) -> Document:
    return HybridExtractor(azure=azure).extract(path, force_ocr=force_ocr)


def boxes_for_range(page: PageText, start: int, end: int) -> list[tuple[float, float, float, float]]:
    return [word.box for word in page.words if word.start < end and start < word.end]
