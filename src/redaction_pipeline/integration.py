from __future__ import annotations

from pathlib import Path

from .pipeline import Pipeline


def run_pipeline_job(
    input_pdf: str | Path,
    output_dir: str | Path,
    policy_path: str | Path,
    **pipeline_options,
) -> dict:
    """Framework-neutral facade for a Celery/Service Bus/DRF background job."""
    result = Pipeline(policy_path, **pipeline_options).run(input_pdf, output_dir)
    return {
        "output_pdf": str(result.output_pdf),
        "audit_json": str(result.audit_json),
        "span_count": result.span_count,
        "verified": result.verified,
    }
