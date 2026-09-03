from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from .models import Span


def load_policy(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        policy = json.load(handle)
    for entity, rule in policy.get("entities", {}).items():
        if rule.get("mode") == "pseudonymize":
            raise ValueError(f"Pseudonymization is not available in the first release ({entity})")
        if rule.get("mode") not in {"mask", "redact", "keep"}:
            raise ValueError(f"Invalid mode for {entity}")
    return policy


def apply_policy(spans: list[Span], policy: dict) -> list[Span]:
    fail_closed = bool(policy.get("fail_closed", True))
    for span in spans:
        rule = policy.get("entities", {}).get(span.entity_type)
        if rule is None:
            if fail_closed:
                span.mode, span.replacement = "redact", None
            else:
                span.mode, span.replacement = "keep", None
        else:
            span.mode = rule["mode"]
            if span.mode == "mask" and rule.get("keep_first_digits") is not None:
                keep = int(rule["keep_first_digits"])
                span.replacement = span.text[:keep] + "*" * max(0, len(span.text) - keep)
            else:
                span.replacement = rule.get("placeholder") if span.mode == "mask" else None
    return spans


def rebuild_pdf(source: str | Path, output: str | Path, spans: list[Span]) -> None:
    """Apply true PDF redactions, strip metadata, and save a rebuilt content stream."""
    with pymupdf.open(source) as pdf:
        for span in spans:
            if span.mode == "keep":
                continue
            page = pdf[span.page - 1]
            for index, coords in enumerate(span.boxes):
                rect = pymupdf.Rect(coords)
                replacement = span.replacement if span.mode == "mask" and index == 0 else None
                page.add_redact_annot(
                    rect,
                    text=replacement,
                    fontname="helv",
                    fontsize=6,
                    fill=(0, 0, 0) if span.mode == "redact" else (1, 1, 1),
                    text_color=(1, 1, 1) if span.mode == "redact" else (0, 0, 0),
                    cross_out=False,
                )
        for page in pdf:
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
        pdf.set_metadata({})
        pdf.del_xml_metadata()
        pdf.save(output, garbage=4, deflate=True, clean=True)
