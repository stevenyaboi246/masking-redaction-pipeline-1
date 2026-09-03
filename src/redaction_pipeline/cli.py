from __future__ import annotations

import argparse
import json

from .pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="De-identify a PDF and emit a PHI-safe audit JSON")
    parser.add_argument("input_pdf")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--policy", default="config/default_policy.json")
    parser.add_argument("--enable-ai", action="store_true")
    parser.add_argument("--enable-azure-language", action="store_true")
    parser.add_argument("--enable-azure-ocr", action="store_true")
    parser.add_argument("--force-ocr", action="store_true", help="OCR every input page with Tesseract")
    args = parser.parse_args()
    result = Pipeline(
        args.policy,
        enable_ai=args.enable_ai,
        enable_azure_language=args.enable_azure_language,
        enable_azure_ocr=args.enable_azure_ocr,
        force_ocr=args.force_ocr,
    ).run(args.input_pdf, args.output_dir)
    print(json.dumps({
        "output_pdf": str(result.output_pdf),
        "audit_json": str(result.audit_json),
        "span_count": result.span_count,
        "verified": result.verified,
    }, indent=2))


if __name__ == "__main__":
    main()
