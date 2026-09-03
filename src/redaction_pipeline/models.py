from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ValidationResult:
    path: str
    size_bytes: int
    page_count: int
    encrypted: bool
    has_metadata: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Word:
    page: int
    text: str
    start: int
    end: int
    box: tuple[float, float, float, float]
    confidence: float = 1.0


@dataclass
class PageText:
    page: int
    text: str
    words: list[Word]
    source: str = "text_layer"
    reliability: float = 1.0


@dataclass
class Document:
    path: str
    pages: list[PageText]


@dataclass
class Span:
    entity_type: str
    page: int
    start: int
    end: int
    text: str
    boxes: list[tuple[float, float, float, float]]
    confidence: float
    engine: str
    mode: Literal["mask", "redact", "keep"] | None = None
    replacement: str | None = None

    def public_dict(self) -> dict:
        value = asdict(self)
        # Audit output must not become a second PHI store.
        value.pop("text", None)
        return value


@dataclass(frozen=True)
class PipelineResult:
    output_pdf: Path
    audit_json: Path
    span_count: int
    verified: bool
