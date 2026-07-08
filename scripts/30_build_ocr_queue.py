#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    FAILED_DIR,
    OCR_QUEUE_DIR,
    OUTPUT_DIR,
    PDF_EXTENSIONS,
    append_jsonl,
    ensure_base_dirs,
    iter_jsonl,
    stable_id,
    write_csv,
    write_json,
    write_jsonl,
)


REGISTRY_JSONL = OUTPUT_DIR / "document_registry.jsonl"
PDF_BACKFILL_CSV = OUTPUT_DIR / "pdf_backfill_candidates.csv"

OUT_QUEUE_JSONL = OCR_QUEUE_DIR / "ocr_queue.jsonl"
OUT_QUEUE_CSV = OCR_QUEUE_DIR / "ocr_queue.csv"
OUT_SUMMARY_JSON = OCR_QUEUE_DIR / "ocr_queue_summary.json"
OUT_QUEUE_FAILURES_JSONL = FAILED_DIR / "ocr_queue_failures.jsonl"


def make_docling_pdf_queue_record(row: Dict[str, Any], reason: str) -> Dict[str, Any]:
    source_path = Path(row["source_path"])

    return {
        **row,
        "queue_id": stable_id(f'docling_surya::{row["variant_id"]}::{source_path.as_posix()}', "dq"),
        "ocr_engine": "docling_surya",
        "ocr_status": "pending",
        "ocr_reason": reason,
        "ocr_input_path": source_path.as_posix(),
        "ocr_input_format": "pdf",
        "original_source_path": source_path.as_posix(),
        "original_source_format": row.get("source_format", "pdf"),
        "from_word_conversion": False,
        "converted_pdf_path": None,
        "pdf_batch_size": 0,
        "defer_vlm": True,
        "agentic_json_path": None,
        "visual_manifest_path": None,
    }


def load_selected_pdfs() -> List[Dict[str, Any]]:
    rows = []

    for row in iter_jsonl(REGISTRY_JSONL):
        source_ext = row.get("source_extension")
        source_format = row.get("source_format")
        source_path = Path(row.get("source_path", ""))

        if source_ext not in PDF_EXTENSIONS and source_format != "pdf":
            continue

        if source_path.exists():
            rows.append(make_docling_pdf_queue_record(row, "selected_pdf_docling_surya"))
        else:
            append_jsonl(
                OUT_QUEUE_FAILURES_JSONL,
                {
                    **row,
                    "error": f"PDF source file not found: {source_path}",
                    "ocr_reason": "selected_pdf_source_missing",
                },
            )

    return rows


def load_pdf_backfill_rows() -> List[Dict[str, Any]]:
    """
    Optional: queue PDF backfill candidates produced by 10_build_registry.py.

    This is disabled by default because the registry already prefers Word when
    Word exists. Use --include-backfill-pdfs only for quality comparison or
    recovery experiments.
    """

    if not PDF_BACKFILL_CSV.exists():
        return []

    import csv

    rows = []

    with PDF_BACKFILL_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            source_path = Path(row.get("source_path", ""))

            if not source_path.exists():
                append_jsonl(
                    OUT_QUEUE_FAILURES_JSONL,
                    {
                        **row,
                        "error": f"Backfill PDF source file not found: {source_path}",
                        "ocr_reason": "backfill_pdf_source_missing",
                    },
                )
                continue

            rows.append(make_docling_pdf_queue_record(row, "backfill_pdf_docling_surya"))

    return rows


def dedupe_queue(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = []
    seen = set()

    for row in rows:
        key = (
            row.get("variant_id"),
            row.get("ocr_input_path"),
            row.get("ocr_reason"),
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(row)

    return deduped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument(
        "--include-backfill-pdfs",
        action="store_true",
        help="Also queue PDFs that were skipped because Word version exists.",
    )
    args = parser.parse_args()

    ensure_base_dirs()
    OCR_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    if not REGISTRY_JSONL.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_JSONL}")

    print("Building Docling + SuryaOCR queue...")

    queue_rows = []

    selected_pdf_rows = load_selected_pdfs()
    queue_rows.extend(selected_pdf_rows)

    backfill_rows = []

    if args.include_backfill_pdfs:
        backfill_rows = load_pdf_backfill_rows()
        queue_rows.extend(backfill_rows)

    queue_rows = dedupe_queue(queue_rows)

    queue_rows = sorted(
        queue_rows,
        key=lambda x: (
            x.get("specialty", ""),
            x.get("disease_path", ""),
            x.get("doc_type", ""),
            x.get("language_hint", ""),
            x.get("source_path", ""),
        ),
    )

    if args.limit and args.limit > 0:
        queue_rows = queue_rows[:args.limit]

    queue_fields = [
        "queue_id",
        "variant_id",
        "disease_class_id",
        "clinical_doc_group_id",
        "specialty",
        "disease_path",
        "doc_type",
        "language_hint",
        "source_format",
        "source_extension",
        "ocr_engine",
        "ocr_status",
        "ocr_reason",
        "ocr_input_path",
        "ocr_input_format",
        "original_source_path",
        "original_source_format",
        "from_word_conversion",
        "converted_pdf_path",
        "size_kb",
    ]

    write_jsonl(OUT_QUEUE_JSONL, queue_rows)
    write_csv(OUT_QUEUE_CSV, queue_rows, queue_fields)

    summary = {
        "queue_total": len(queue_rows),
        "selected_pdf_total": len(selected_pdf_rows),
        "backfill_pdf_total": len(backfill_rows),
        "by_language_hint": dict(Counter(x.get("language_hint", "unknown") for x in queue_rows)),
        "by_doc_type": dict(Counter(x.get("doc_type", "unknown") for x in queue_rows)),
        "by_ocr_reason": dict(Counter(x.get("ocr_reason", "unknown") for x in queue_rows)),
        "outputs": {
            "ocr_queue_jsonl": OUT_QUEUE_JSONL.as_posix(),
            "ocr_queue_csv": OUT_QUEUE_CSV.as_posix(),
            "queue_failures_jsonl": OUT_QUEUE_FAILURES_JSONL.as_posix(),
            "summary_json": OUT_SUMMARY_JSON.as_posix(),
        },
    }

    write_json(OUT_SUMMARY_JSON, summary)

    print("\nDONE")

    for k, v in summary.items():
        if k != "outputs":
            print(f"{k}: {v}")

    print("\nSaved:")

    for _, v in summary["outputs"].items():
        print(v)


if __name__ == "__main__":
    main()
