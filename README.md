# Medical de-identification pipeline demo

Runnable Python demo implementing the uploaded design's path from validation through rebuilt PDF delivery:

1. Validate, hash, and structurally inspect the PDF.
2. Parse each reliable text layer with PyMuPDF and retain word geometry.
3. OCR weak or image-only pages with local Tesseract. Azure Document Intelligence is an optional last fallback.
4. Detect PII/PHI with deterministic rules, optional Azure AI Language, and optional Claude.
5. Merge overlaps by confidence and apply the project's per-entity policy.
6. Use PyMuPDF redaction annotations to remove text or image pixels and insert masking tokens.
7. Save a cleaned, rebuilt PDF with metadata removed.
8. Force Tesseract OCR over the rebuilt visual output, rerun enabled detectors, and search for leaked originals. Failed verification deletes the deliverable.
9. Write a PHI-safe audit JSON that contains geometry and provenance but not removed values.

The package is framework-neutral so another developer can call `Pipeline.run()` from a Django REST Framework worker without coupling document processing to an HTTP request.

## Local setup

Install the Tesseract binary first:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

Then install the Python project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ai,azure]'
pytest
```

## Run the local demo

This command uses PyMuPDF plus Tesseract and the rule detector. It requires no cloud credentials:

```bash
redact-pdf path/to/chart.pdf --output-dir output
```

Generate and process the included synthetic example:

```bash
python scripts/create_demo_pdf.py
redact-pdf examples/synthetic-chart.pdf --output-dir examples/output
```

Force Tesseract on every input page, useful for testing scans:

```bash
redact-pdf path/to/chart.pdf --output-dir output --force-ocr
```

## Enable Claude detection

```bash
export ANTHROPIC_API_KEY='...'
export ANTHROPIC_MODEL='your-approved-model-id'
redact-pdf chart.pdf --output-dir output --enable-ai
```

Claude handles contextual categories such as patient, physician, guarantor, facility, and employer names. Exact model-returned text is mapped back to locally extracted words so every span has PDF geometry. Never send PHI to an account or endpoint that is outside the approved trust boundary.

## Enable Azure adapters

```bash
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT='https://...'
export AZURE_DOCUMENT_INTELLIGENCE_KEY='...'
export AZURE_LANGUAGE_ENDPOINT='https://...'
export AZURE_LANGUAGE_KEY='...'

redact-pdf chart.pdf --output-dir output \
  --enable-azure-ocr \
  --enable-azure-language \
  --enable-ai
```

Azure OCR is not the normal local path: it is consulted only when the text layer is unreliable and Tesseract remains below the reliability threshold. In production, replace API-key construction with Managed Identity and private endpoints.

## Integration contract for DRF/React

Call the pipeline from a background worker, not directly in a request handler:

```python
from redaction_pipeline import Pipeline

pipeline = Pipeline(
    policy_path="config/default_policy.json",
    enable_ai=True,
    enable_azure_language=True,
    enable_azure_ocr=True,
)
result = pipeline.run("/secure/jobs/input.pdf", "/secure/jobs/output")
```

Return or persist these fields after the worker succeeds:

```json
{
  "output_pdf": "...deidentified.pdf",
  "audit_json": "...audit.json",
  "span_count": 10,
  "verified": true
}
```

The backend remains responsible for authentication, project/config lookup, storage adapters, job status, queueing, retention, and authorization. React should receive job/result metadata from DRF, never local filesystem paths.

## Demo limits

- Cloud calls cannot be integration-tested without project credentials; offline adapter tests validate response mapping and geometry.
- Tesseract OCR quality depends on scan resolution, rotation, handwriting, and language packs.
- The demo implements masking/redaction only. A `pseudonymize` policy is rejected as required for the first release.
- Before production, add a golden medical-chart dataset, per-entity precision/recall gates, rotated-page tests, and a human-review path.
