#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.common import (
    CONVERTED_DIR,
    FAILED_DIR,
    LIBREOFFICE_CMD,
    OCR_QUEUE_DIR,
    OUTPUT_DIR,
    PDF_EXTENSIONS,
    WORD_EXTENSIONS,
    append_jsonl,
    ensure_base_dirs,
    iter_jsonl,
    read_jsonl,
    run_cmd,
    stable_id,
    write_csv,
    write_json,
    write_jsonl,
)


REGISTRY_JSONL = OUTPUT_DIR / "document_registry.jsonl"
WORD_FAILURES_JSONL = FAILED_DIR / "word_failures.jsonl"

OUT_QUEUE_JSONL = OCR_QUEUE_DIR / "ocr_queue.jsonl"
OUT_QUEUE_CSV = OCR_QUEUE_DIR / "ocr_queue.csv"
OUT_SUMMARY_JSON = OCR_QUEUE_DIR / "ocr_queue_summary.json"
OUT_CONVERSION_FAILURES_JSONL = FAILED_DIR / "ocr_queue_conversion_failures.jsonl"


def libreoffice_pdf_convert_isolated(source_path: Path, variant_id: str) -> Path:
    """
    Convert DOC/DOCX to PDF for PaddleOCR-VL.
    Uses isolated LibreOffice profile to reduce parallel conversion conflicts.
    """
    out_dir = CONVERTED_DIR / "word_to_pdf" / variant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_dir = (CONVERTED_DIR / "lo_profiles_pdf" / variant_id).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_uri = profile_dir.as_uri()

    cmd = [
        LIBREOFFICE_CMD,
        f"-env:UserInstallation={profile_uri}",
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(source_path),
    ]

    run_cmd(cmd, timeout=360)

    expected = out_dir / f"{source_path.stem}.pdf"
    if expected.exists():
        return expected

    matches = list(out_dir.glob(f"{source_path.stem}*.pdf"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"LibreOffice PDF output not found: {source_path}")


def make_pdf_queue_record(row: Dict[str, Any], reason: str) -> Dict[str, Any]:
    source_path = Path(row["source_path"])

    return {
        **row,
        "queue_id": stable_id(f'ocr::{row["variant_id"]}::{source_path.as_posix()}', "oq"),
        "ocr_engine": "paddleocr_vl",
        "ocr_status": "pending",
        "ocr_reason": reason,
        "ocr_input_path": source_path.as_posix(),
        "ocr_input_format": "pdf",
        "original_source_path": source_path.as_posix(),
        "original_source_format": row.get("source_format", "pdf"),
        "from_word_conversion": False,
        "converted_pdf_path": None,
    }


def make_word_conversion_queue_record(row: Dict[str, Any], converted_pdf_path: Path, reason: str) -> Dict[str, Any]:
    source_path = Path(row["source_path"])

    return {
        **row,
        "queue_id": stable_id(f'ocr::{row["variant_id"]}::{converted_pdf_path.as_posix()}', "oq"),
        "ocr_engine": "paddleocr_vl",
        "ocr_status": "pending",
        "ocr_reason": reason,
        "ocr_input_path": converted_pdf_path.as_posix(),
        "ocr_input_format": "pdf",
        "original_source_path": source_path.as_posix(),
        "original_source_format": row.get("source_format"),
        "from_word_conversion": True,
        "converted_pdf_path": converted_pdf_path.as_posix(),
    }


def load_selected_pdfs() -> List[Dict[str, Any]]:
    rows = []

    for row in iter_jsonl(REGISTRY_JSONL):
        if row.get("source_extension") in PDF_EXTENSIONS or row.get("source_format") == "pdf":
            source_path = Path(row["source_path"])
            if source_path.exists():
                rows.append(make_pdf_queue_record(row, "selected_pdf_primary_ocr"))
            else:
                append_jsonl(OUT_CONVERSION_FAILURES_JSONL, {
                    **row,
                    "error": f"PDF source file not found: {source_path}",
                    "ocr_reason": "selected_pdf_source_missing",
                })

    return rows


def load_word_failures() -> List[Dict[str, Any]]:
    if not WORD_FAILURES_JSONL.exists():
        return []

    rows = []
    seen = set()

    for row in iter_jsonl(WORD_FAILURES_JSONL):
        variant_id = row.get("variant_id")
        source_path = row.get("source_path")

        if not variant_id or not source_path:
            continue

        key = (variant_id, source_path)
        if key in seen:
            continue
        seen.add(key)

        source_format = row.get("source_format")
        source_ext = row.get("source_extension") or Path(source_path).suffix.lower()

        if source_format in {"docx", "doc"} or source_ext in WORD_EXTENSIONS:
            rows.append(row)

    return rows


def convert_word_failure_worker(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    try:
        source_path = Path(row["source_path"])

        if not source_path.exists():
            raise FileNotFoundError(f"Word source file not found: {source_path}")

        converted_pdf = libreoffice_pdf_convert_isolated(source_path, row["variant_id"])

        reason = row.get("ocr_reason") or "word_native_extraction_failed_or_weak"
        queue_record = make_word_conversion_queue_record(row, converted_pdf, reason)

        return "ok", queue_record

    except Exception as e:
        failed = {
            **row,
            "ocr_status": "conversion_failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        return "failed", failed


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
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--word-failures-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_base_dirs()
    OCR_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    print("Building OCR queue...")

    queue_rows = []
    conversion_failures = 0

    # 1. Selected PDFs from registry
    if not args.word_failures_only:
        pdf_rows = load_selected_pdfs()
        queue_rows.extend(pdf_rows)
        print(f"Selected PDFs queued: {len(pdf_rows)}")

    # 2. Failed/weak Word files → convert to PDF
    word_failure_rows = []

    if not args.pdf_only:
        word_failure_rows = load_word_failures()

        if args.limit > 0:
            word_failure_rows = word_failure_rows[:args.limit]

        print(f"Word failures to convert: {len(word_failure_rows)}")
        print(f"LibreOffice workers: {args.workers}")

        if word_failure_rows:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(convert_word_failure_worker, row)
                    for row in word_failure_rows
                ]

                for future in tqdm(as_completed(futures), total=len(futures), desc="Converting Word failures to PDF"):
                    status, result = future.result()

                    if status == "ok":
                        queue_rows.append(result)
                    else:
                        conversion_failures += 1
                        append_jsonl(OUT_CONVERSION_FAILURES_JSONL, result)

    queue_rows = dedupe_queue(queue_rows)

    queue_rows = sorted(
        queue_rows,
        key=lambda x: (
            x.get("from_word_conversion", False),
            x.get("specialty", ""),
            x.get("disease_path", ""),
            x.get("doc_type", ""),
            x.get("source_path", ""),
        )
    )

    if args.limit > 0 and args.pdf_only:
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
        "pdf_primary_total": sum(1 for x in queue_rows if not x.get("from_word_conversion")),
        "word_converted_total": sum(1 for x in queue_rows if x.get("from_word_conversion")),
        "word_failures_seen": len(word_failure_rows),
        "word_conversion_failures": conversion_failures,
        "outputs": {
            "ocr_queue_jsonl": OUT_QUEUE_JSONL.as_posix(),
            "ocr_queue_csv": OUT_QUEUE_CSV.as_posix(),
            "conversion_failures_jsonl": OUT_CONVERSION_FAILURES_JSONL.as_posix(),
            "summary_json": OUT_SUMMARY_JSON.as_posix(),
        }
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
