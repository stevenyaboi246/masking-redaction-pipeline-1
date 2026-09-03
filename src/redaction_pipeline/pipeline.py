from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .detection import AnthropicDetector, AzureLanguageDetector, RuleDetector, resolve_spans
from .extraction import AzureDocumentIntelligenceOCR, HybridExtractor
from .models import PipelineResult, Span
from .transform import apply_policy, load_policy, rebuild_pdf
from .validation import validate_pdf


class VerificationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _boxes_overlap(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> bool:
        return not (
            first[2] <= second[0]
            or second[2] <= first[0]
            or first[3] <= second[1]
            or second[3] <= first[1]
        )

class Pipeline:
    def __init__(self, policy_path: str | Path, enable_ai: bool = False,
                 enable_azure_language: bool = False, enable_azure_ocr: bool = False,
                 force_ocr: bool = False):
        self.policy = load_policy(policy_path)
        self.force_ocr = force_ocr
        self.detectors = [RuleDetector()]
        if enable_ai:
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError("--enable-ai requires ANTHROPIC_API_KEY")
            self.detectors.append(AnthropicDetector(key, os.environ.get("ANTHROPIC_MODEL", "")))
        azure_ocr = None
        if enable_azure_language or enable_azure_ocr:
            try:
                from azure.core.credentials import AzureKeyCredential
            except ImportError as exc:
                raise RuntimeError("Install Azure support with: pip install -e '.[azure]'") from exc
        if enable_azure_language:
            endpoint, key = os.environ.get("AZURE_LANGUAGE_ENDPOINT", ""), os.environ.get("AZURE_LANGUAGE_KEY", "")
            if not endpoint or not key:
                raise ValueError("Azure Language requires AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY")
            self.detectors.append(AzureLanguageDetector(endpoint, AzureKeyCredential(key)))
        if enable_azure_ocr:
            endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
            key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
            if not endpoint or not key:
                raise ValueError("Azure OCR requires AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY")
            azure_ocr = AzureDocumentIntelligenceOCR(endpoint, AzureKeyCredential(key))
        self.extractor = HybridExtractor(azure=azure_ocr)

    def run(self, source: str | Path, output_dir: str | Path) -> PipelineResult:
        source = Path(source)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        validation = validate_pdf(source)
        document = self.extractor.extract(source, force_ocr=self.force_ocr)
        raw: list[Span] = []
        for detector in self.detectors:
            raw.extend(detector.detect(document))
        spans = apply_policy(resolve_spans(raw, float(self.policy.get("min_confidence", 0.3))), self.policy)

        output_pdf = out_dir / f"{source.stem}.deidentified.pdf"
        audit_json = out_dir / f"{source.stem}.audit.json"
        rebuild_pdf(source, output_pdf, spans)

        try:
            verified, remaining_count, verification_findings = self._verify(output_pdf, spans)
        except Exception:
            # Never leave an apparently valid but unverified result behind.
            output_pdf.unlink(missing_ok=True)
            audit_json.unlink(missing_ok=True)
            raise
        audit = {
            "schema_version": "1.0",
            "job_id": job_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": _sha256(source),
            "output_sha256": _sha256(output_pdf),
            "validation": asdict(validation),
            "extraction": [{"page": p.page, "engine": p.source, "reliability": round(p.reliability, 4)} for p in document.pages],
            "counts": {
                "found": len(spans),
                "masked": sum(s.mode == "mask" for s in spans),
                "redacted": sum(s.mode == "redact" for s in spans),
                "kept": sum(s.mode == "keep" for s in spans),
            },
            "spans": [span.public_dict() for span in spans],
            "verification": {"passed": verified, "remaining_count": remaining_count, "findings": verification_findings},
        }
        audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        if self.policy.get("verify", True) and not verified:
            output_pdf.unlink(missing_ok=True)
            raise VerificationError(f"Verification failed: {remaining_count} identifier(s) remain; output PDF removed")
        return PipelineResult(output_pdf, audit_json, len(spans), verified)

    def _verify(self, output: Path, original_spans: list[Span],) -> tuple[bool, int, list[dict]]:
        """
        Run detection again on the rebuilt PDF.

        Returns:
           passed:
               True when no identifiers remain.

          remaining_count:
               Number of verification findings.

        findings:
            PHI-safe information about remaining identifiers.
            Original text values are intentionally excluded.
        """

        # Force OCR so verification checks what is visually present,
        # including scanned/image-based pages.
        rebuilt = self.extractor.extract(
            output,
            force_ocr=True,
            allow_blank=True,
        )

        # Run every enabled detector again.
        detected_again: list[Span] = []

        for detector in self.detectors:
            detected_again.extend(
                detector.detect(rebuilt)
            )

        # Resolve overlapping detections and apply the same policy.
        remaining = resolve_spans(
            detected_again,
            float(self.policy.get("min_confidence", 0.3)),
        )

        remaining = apply_policy(
            remaining,
            self.policy,
        )

        # Values configured as "keep" are allowed to remain.
        remaining = [
            span
            for span in remaining
            if span.mode != "keep"
        ]

        # Collect the areas occupied by masking placeholders.
        masked_boxes_by_page: dict[
            int,
            list[tuple[float, float, float, float]],
        ] = {}

        for original_span in original_spans:
            if original_span.mode != "mask":
                continue

            masked_boxes_by_page.setdefault(
                original_span.page,
                [],
            ).extend(original_span.boxes)

        def is_mask_placeholder_artifact(span: Span) -> bool:
            """
            Return True when the second-pass detection is completely
            contained in an area where a masking placeholder was inserted.
            """

            masked_boxes = masked_boxes_by_page.get(span.page,[],
    )

            if not span.boxes or not masked_boxes:
                return False

            return all(
                any(
                    _boxes_overlap(
                        detected_box,
                        masked_box,
                    )
                    for masked_box in masked_boxes
                )
                for detected_box in span.boxes
            )

        # Ignore Claude/Tesseract detections produced by placeholders such
        # as [FACILITY], [PERSON_NAME], and [MRN-1].
        remaining = [
            span
            for span in remaining
            if not is_mask_placeholder_artifact(span)
        ]

        # Check whether original values still appear in the rebuilt OCR text.
        rebuilt_text = "\n".join(
            page.text
            for page in rebuilt.pages
        )

        leaked_originals = [
            span
            for span in original_spans
            if (
                span.mode != "keep"
                and span.text
                and span.text.casefold() in rebuilt_text.casefold()
            )
        ]

        # Record findings without recording the original PHI values.
        findings: list[dict] = []

        for span in remaining:
            findings.append(
                {
                    "reason": "detected_on_second_pass",
                    "entity_type": span.entity_type,
                    "page": span.page,
                    "confidence": span.confidence,
                    "engine": span.engine,
                    "boxes": span.boxes,
                }
            )

        for span in leaked_originals:
            findings.append(
                {
                    "reason": "original_value_survived",
                    "entity_type": span.entity_type,
                    "page": span.page,
                    "confidence": span.confidence,
                    "engine": span.engine,
                    "boxes": span.boxes,
                }
            )

        remaining_count = len(findings)
        passed = remaining_count == 0

        return passed, remaining_count, findings
    