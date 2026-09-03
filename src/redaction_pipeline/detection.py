from __future__ import annotations

import json
import os
from pyexpat import model
import re
from typing import Iterable

from .extraction import boxes_for_range
from .models import Document, PageText, Span


PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("contact", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("contact", re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("payment_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("other_date", re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b")),
    ("zip_code", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
    ("mrn", re.compile(r"(?i)\b(?:MRN|medical record(?: number)?)[\s:#-]*((?=[A-Z0-9-]*\d)[A-Z0-9-]{4,20})\b")),
    ("member_id", re.compile(r"(?i)\bmember(?:\s+ID)?[\s:#-]*((?=[A-Z0-9-]*\d)[A-Z0-9-]{4,20})\b")),
    ("age", re.compile(r"(?i)\bage[\s:]+(\d{1,3})\b")),
]

MASK_PLACEHOLDER_TOKENS = {
    "PATIENT_NAME",
    "PHYSICIAN",
    "PERSON_NAME",
    "FACILITY",
    "EMPLOYER",
    "GUARANTOR",
    "ADDRESS",
    "DOB",
    "DATE",
    "MRN",
    "MRN-1",
    "MEMBER_ID",
    "AGE_90_PLUS",
}


def _span(page: PageText, entity: str, start: int, end: int, confidence: float, engine: str) -> Span | None:
    boxes = boxes_for_range(page, start, end)
    if not boxes:
        return None
    return Span(entity, page.page, start, end, page.text[start:end], boxes, confidence, engine)


class RuleDetector:
    def detect(self, document: Document) -> list[Span]:
        found: list[Span] = []
        for page in document.pages:
            for entity, pattern in PATTERNS:
                for match in pattern.finditer(page.text):
                    start, end = match.span(1) if match.lastindex else match.span()
                    resolved_entity = entity
                    context = page.text[max(0, start - 40):start].lower()
                    if entity == "other_date":
                        if re.search(r"(?:dob|date of birth)[\s:#-]*$", context):
                            resolved_entity = "date_of_birth"
                        elif re.search(r"(?:dos|date of service|service date)[\s:#-]*$", context):
                            resolved_entity = "date_of_service"
                    elif entity == "age":
                        resolved_entity = "age_over_89" if int(match.group(1)) > 89 else "age_89_or_below"
                    candidate = _span(page, resolved_entity, start, end, 0.99, "rule")
                    if candidate:
                        found.append(candidate)
        return found

def parse_claude_items(raw: str) -> list[dict]:
    """Extract JSON arrays even if Claude adds fences, prose, or corrections."""
    decoder = json.JSONDecoder()
    items: list[dict] = []
    found_array = False

    # Prefer fenced blocks when Claude uses Markdown.
    fenced_blocks = re.findall(
        r"```(?:json)?\s*(.*?)```",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )

    candidates = fenced_blocks or [raw]

    for candidate in candidates:
        position = 0

        while position < len(candidate):
            array_start = candidate.find("[", position)

            if array_start == -1:
                break

            try:
                value, consumed = decoder.raw_decode(
                    candidate[array_start:]
                )
            except json.JSONDecodeError:
                position = array_start + 1
                continue

            position = array_start + consumed

            if isinstance(value, list):
                found_array = True
                items.extend(
                    item for item in value
                    if isinstance(item, dict)
                )

    if not found_array:
        # Do not include raw in the exception because it may contain PHI.
        raise RuntimeError(
            "Claude response did not contain a valid JSON array"
        )

    return items

class AnthropicDetector:
    """Context detector. Exact quoted values are mapped back to local PDF geometry."""

    def __init__(self, api_key: str, model: str):
        if not model:
            raise ValueError(
                "ANTHROPIC_MODEL must be set when AI detection is enabled"
            )

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Install AI support with: pip install -e '.[ai]'"
            ) from exc

        workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()

        client_options = {
            "api_key": api_key,
        }

        if workspace_id:
            client_options["default_headers"] = {
                "anthropic-workspace-id": workspace_id
            }

        self.client = Anthropic(**client_options)
        self.model = model

    def detect(self, document: Document) -> list[Span]:
        found: list[Span] = []
        for page in document.pages:
            prompt = (
                "Find every occurrence of personally identifying information in this "
                "authorized medical document page. Prioritize recall over precision. "

                "Include patient names, physicians, surgeons, nurses, relatives and "
                "family-history names, guarantors, emergency contacts, employees named "
                "in 'Printed by' or similar fields, facilities, employers, street "
                "addresses, and dates of birth. Find repeated occurrences, including "
                "standalone surnames. "

                "Do not return medical conditions, medications, procedures, or ordinary ages"
                "Never return masking placeholders or placeholder fragments, including "
                "PATIENT_NAME, PHYSICIAN, PERSON_NAME, FACILITY, EMPLOYER, "
                "GUARANTOR, ADDRESS, DOB, DATE, MRN, MEMBER_ID, or AGE_90_PLUS. "

                "Return a JSON array only. Each object must have exactly these keys: "
                "entity_type, text, confidence. Copy text exactly from PAGE TEXT. "
                "Do not infer or correct OCR text. "

                "Allowed entity_type values: patient_name, physician_name, person_name, "
                "facility_name, employer, street_address, date_of_birth.\n\n"

                f"PAGE TEXT:\n{page.text}"
            )
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
            # raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
            for item in parse_claude_items(raw):
                value = str(item.get("text", ""))
                entity = str(item.get("entity_type", ""))
                name_like_entities = {
                    "patient_name",
                    "physician_name",
                    "person_name",
                    "facility_name",
                    "employer",
                    "street_address",
                }

                if entity in name_like_entities:
                    # Names, organizations, and addresses must contain letters.
                    if len(value.strip()) < 2 or not re.search(r"[A-Za-z]", value):
                        continue
                normalized_value = re.sub(
                    r"[^A-Z0-9_-]",
                    "",
                    value.upper(),
                )
                if normalized_value in MASK_PLACEHOLDER_TOKENS:
                    continue
                if not value or entity not in {"patient_name", "physician_name", "person_name", "facility_name", "employer", "street_address", "date_of_birth"}:
                    continue
                for match in re.finditer(re.escape(value), page.text):
                    candidate = _span(page, entity, *match.span(), float(item.get("confidence", 0.85)), "anthropic")
                    if candidate:
                        found.append(candidate)
        return found


class AzureLanguageDetector:
    engine_name = "azure_ai_language"
    CATEGORY_MAP = {
        "PhoneNumber": "contact",
        "Email": "contact",
        "URL": "contact",
        "Address": "street_address",
        "USSocialSecurityNumber": "ssn",
        "CreditCardNumber": "payment_card",
        "IPAddress": "ip_address",
        "Person": "person_name",
        "Organization": "facility_name",
    }

    def __init__(self, endpoint: str, credential):
        try:
            from azure.ai.textanalytics import TextAnalyticsClient
        except ImportError as exc:
            raise RuntimeError("Install Azure support with: pip install -e '.[azure]'") from exc
        self.client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

    def detect(self, document: Document) -> list[Span]:
        found: list[Span] = []
        for page in document.pages:
            result = self.client.recognize_pii_entities([page.text])[0]
            if result.is_error:
                raise RuntimeError(f"Azure Language failed on page {page.page}: {result.message}")
            for entity in result.entities:
                mapped = self.CATEGORY_MAP.get(entity.category)
                if not mapped:
                    continue
                candidate = _span(page, mapped, int(entity.offset), int(entity.offset + entity.length), float(entity.confidence_score), self.engine_name)
                if candidate:
                    found.append(candidate)
        return found


def resolve_spans(spans: Iterable[Span], min_confidence: float) -> list[Span]:
    candidates = sorted((s for s in spans if s.confidence >= min_confidence), key=lambda s: (s.page, s.start, -s.confidence, -(s.end - s.start)))
    resolved: list[Span] = []
    for span in candidates:
        collisions = [i for i, old in enumerate(resolved) if old.page == span.page and old.start < span.end and span.start < old.end]
        if not collisions:
            resolved.append(span)
            continue
        best = max([span, *(resolved[i] for i in collisions)], key=lambda s: (s.confidence, s.end - s.start))
        for i in reversed(collisions):
            resolved.pop(i)
        resolved.append(best)
    return sorted(resolved, key=lambda s: (s.page, s.start))
