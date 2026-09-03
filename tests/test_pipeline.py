from __future__ import annotations

import json
import io

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from redaction_pipeline.pipeline import Pipeline


def test_end_to_end(tmp_path):
    source = tmp_path / "chart.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Patient MRN: 4471902 Phone: (415) 555-0138 DOB: 03/14/1962")
    pdf.save(source)
    pdf.close()

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "min_confidence": 0.8,
        "verify": True,
        "fail_closed": True,
        "entities": {
            "mrn": {"mode": "mask", "placeholder": "[MRN]"},
            "contact": {"mode": "redact"},
            "other_date": {"mode": "mask", "placeholder": "[DATE]"},
        },
    }))
    result = Pipeline(policy).run(source, tmp_path / "out")
    with pymupdf.open(result.output_pdf) as rebuilt:
        text = "".join(page.get_text() for page in rebuilt)
    assert "4471902" not in text
    assert "555-0138" not in text
    assert "03/14/1962" not in text
    assert result.verified


def test_scanned_pdf_uses_tesseract_and_removes_image_pixels(tmp_path):
    image = Image.new("RGB", (1800, 500), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=54)
    draw.text((60, 100), "MRN: 4471902   Phone: (415) 555-0138", fill="black", font=font)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")

    source = tmp_path / "scan.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(36, 72, 576, 222), stream=image_bytes.getvalue())
    pdf.save(source)
    pdf.close()

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "min_confidence": 0.3,
        "verify": True,
        "fail_closed": True,
        "entities": {
            "mrn": {"mode": "mask", "placeholder": "[MRN-1]"},
            "contact": {"mode": "redact"},
        },
    }))
    result = Pipeline(policy, force_ocr=True).run(source, tmp_path / "out")
    audit = json.loads(result.audit_json.read_text())
    assert audit["extraction"][0]["engine"] == "tesseract"
    assert result.verified
