#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Must be set before Docling / torch-heavy imports.
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("DOCLING_NUM_THREADS", "1"))
os.environ.setdefault("DOCLING_NUM_THREADS", os.getenv("DOCLING_NUM_THREADS", "1"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.common import (
    FAILED_DIR,
    OCR_QUEUE_DIR,
    OUTPUT_DIR,
    PARSED_DOCLING_DIR,
    PARSED_OCR_DIR,
    append_jsonl,
    clean_text,
    detect_language_from_text,
    ensure_base_dirs,
    get_quality_flags,
    iter_jsonl,
    write_json,
    write_jsonl,
)


QUEUE_JSONL = OCR_QUEUE_DIR / "ocr_queue.jsonl"

OUT_PARSED_DOCLING_JSONL = OUTPUT_DIR / "parsed_docling.jsonl"
OUT_PARSED_OCR_JSONL = OUTPUT_DIR / "parsed_ocr.jsonl"  # compatibility alias
OUT_SUMMARY_JSON = OUTPUT_DIR / "parsed_docling_summary.json"
FAILED_JSONL = FAILED_DIR / "docling_surya_failures.jsonl"


# ============================================================
# Env config
# ============================================================

def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)

    if raw is None:
        return default

    try:
        return float(raw)
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)

    if raw is None:
        return default

    try:
        return int(raw)
    except Exception:
        return default


def get_ocr_langs() -> List[str]:
    raw = os.getenv("DOCLING_OCR_LANGS", "uz,ru,en")
    langs = [x.strip() for x in raw.split(",") if x.strip()]
    return langs or ["uz", "ru", "en"]


# ============================================================
# Docling converter
# ============================================================

def build_docling_pdf_converter():
    """
    Stable PDF converter for Docling + SuryaOCR.

    Important settings come from the working local test:
      - PyPdfiumDocumentBackend
      - SuryaOcrOptions(lang=["uz", "ru", "en"], force_full_page_ocr=True)
      - num_threads=1
      - no page/picture image export
      - table structure disabled by default

    This mirrors your confirmed working extract_test.py config. :contentReference[oaicite:1]{index=1}
    """

    from docling_surya import SuryaOcrOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()

    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = SuryaOcrOptions(
        lang=get_ocr_langs(),
        force_full_page_ocr=env_bool("DOCLING_FORCE_FULL_PAGE_OCR", True),
    )

    # Safer default. Enable later only if needed.
    pipeline_options.do_table_structure = env_bool("DOCLING_DO_TABLE_STRUCTURE", False)

    # Do not save embedded page/picture images.
    pipeline_options.generate_page_images = env_bool("DOCLING_GENERATE_PAGE_IMAGES", False)
    pipeline_options.generate_picture_images = env_bool("DOCLING_GENERATE_PICTURE_IMAGES", False)
    pipeline_options.images_scale = env_float("DOCLING_IMAGES_SCALE", 1.0)

    # Required for docling-surya.
    pipeline_options.allow_external_plugins = True

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=env_int("DOCLING_NUM_THREADS", 1),
        device=AcceleratorDevice.AUTO,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    return converter


# ============================================================
# Record helpers
# ============================================================

def count_docling_items(docling_dict: Dict[str, Any]) -> Dict[str, int]:
    return {
        "texts": len(docling_dict.get("texts") or []),
        "tables": len(docling_dict.get("tables") or []),
        "pictures": len(docling_dict.get("pictures") or []),
        "groups": len(docling_dict.get("groups") or []),
        "pages": len(docling_dict.get("pages") or []),
    }


def write_pdf_outputs(
    variant_dir: Path,
    record: Dict[str, Any],
    markdown_content: str,
    docling_dict: Dict[str, Any],
) -> None:
    """
    Writes JSON + Markdown only:
      data/parsed_docling/{variant_id}/parsed.json
      data/parsed_docling/{variant_id}/document.docling.json
      data/parsed_docling/{variant_id}/document.md

    No TXT output.
    """

    variant_dir.mkdir(parents=True, exist_ok=True)

    md_path = variant_dir / "document.md"
    docling_json_path = variant_dir / "document.docling.json"
    parsed_json_path = variant_dir / "parsed.json"

    md_path.write_text(markdown_content, encoding="utf-8")
    docling_json_path.write_text(
        json.dumps(docling_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    record["markdown_path"] = md_path.as_posix()
    record["docling_json_path"] = docling_json_path.as_posix()

    write_json(parsed_json_path, record)


# ============================================================
# Extraction
# ============================================================

def run_docling_surya_one(
    converter: Any,
    queue_row: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    variant_id = queue_row["variant_id"]
    input_path = Path(queue_row["ocr_input_path"])

    if not input_path.exists():
        raise FileNotFoundError(f"Docling input PDF not found: {input_path}")

    variant_dir = PARSED_DOCLING_DIR / variant_id
    final_json_path = variant_dir / "parsed.json"

    if final_json_path.exists() and not force:
        return json.loads(final_json_path.read_text(encoding="utf-8"))

    if variant_dir.exists() and force:
        import shutil

        shutil.rmtree(variant_dir)

    variant_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    result = converter.convert(str(input_path))
    doc = result.document

    markdown_content = doc.export_to_markdown()
    raw_text = clean_text(doc.export_to_text())
    docling_dict = doc.export_to_dict()

    detected_language = detect_language_from_text(raw_text)
    quality_flags = get_quality_flags(raw_text, detected_language)

    item_counts = count_docling_items(docling_dict)
    elapsed_seconds = round(time.time() - started, 3)

    record = {
        **queue_row,
        "detected_language": detected_language,
        "extraction_method": "docling_surya_pdf_pypdfium2",
        "raw_output_dir": variant_dir.as_posix(),
        "char_count": len(raw_text),
        "quality_flags": quality_flags,
        "elapsed_seconds": elapsed_seconds,
        "docling_counts": item_counts,
        "raw_text": raw_text,
    }

    write_pdf_outputs(
        variant_dir=variant_dir,
        record=record,
        markdown_content=markdown_content,
        docling_dict=docling_dict,
    )

    return record


def rebuild_parsed_docling_jsonl() -> int:
    rows = []

    for p in sorted(PARSED_DOCLING_DIR.glob("*/parsed.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass

    write_jsonl(OUT_PARSED_DOCLING_JSONL, rows)

    # Compatibility with older merge script names.
    write_jsonl(OUT_PARSED_OCR_JSONL, rows)

    return len(rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-backfill", action="store_true")
    parser.add_argument("--only-primary", action="store_true")
    args = parser.parse_args()

    ensure_base_dirs()
    PARSED_DOCLING_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_OCR_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(QUEUE_JSONL))

    if args.only_backfill:
        rows = [r for r in rows if str(r.get("ocr_reason", "")).startswith("backfill")]

    if args.only_primary:
        rows = [r for r in rows if not str(r.get("ocr_reason", "")).startswith("backfill")]

    rows = rows[args.start:]

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    if not rows:
        print("No Docling OCR queue rows found.")
        return

    print(f"Docling + SuryaOCR rows selected: {len(rows)}")
    print(f"OCR langs: {get_ocr_langs()}")
    print(f"Force full-page OCR: {env_bool('DOCLING_FORCE_FULL_PAGE_OCR', True)}")
    print(f"Table structure: {env_bool('DOCLING_DO_TABLE_STRUCTURE', False)}")
    print(f"Docling threads: {env_int('DOCLING_NUM_THREADS', 1)}")
    print(f"Surya backend: {os.getenv('SURYA_INFERENCE_BACKEND', 'vllm')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '')}")

    converter = build_docling_pdf_converter()

    parsed_ok = 0
    failed = 0
    skipped_existing = 0
    total_chars = 0
    total_seconds = 0.0

    for row in tqdm(rows, desc="Docling+Surya"):
        final_json_path = PARSED_DOCLING_DIR / row["variant_id"] / "parsed.json"

        if final_json_path.exists() and not args.force:
            skipped_existing += 1
            continue

        try:
            result = run_docling_surya_one(converter, row, force=args.force)

            parsed_ok += 1
            total_chars += int(result.get("char_count") or 0)
            total_seconds += float(result.get("elapsed_seconds") or 0)

        except Exception as e:
            failed += 1

            append_jsonl(
                FAILED_JSONL,
                {
                    **row,
                    "ocr_status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

        finally:
            gc.collect()

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    total_parsed_files = rebuild_parsed_docling_jsonl()

    summary = {
        "ocr_rows_considered": len(rows),
        "parsed_ok": parsed_ok,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "total_parsed_docling_files": total_parsed_files,
        "total_chars_current_run": total_chars,
        "total_seconds_current_run": round(total_seconds, 3),
        "avg_seconds_per_success": round(total_seconds / parsed_ok, 3) if parsed_ok else None,
        "ocr_engine": "docling_surya",
        "ocr_langs": get_ocr_langs(),
        "force_full_page_ocr": env_bool("DOCLING_FORCE_FULL_PAGE_OCR", True),
        "do_table_structure": env_bool("DOCLING_DO_TABLE_STRUCTURE", False),
        "docling_num_threads": env_int("DOCLING_NUM_THREADS", 1),
        "surya_backend": os.getenv("SURYA_INFERENCE_BACKEND", "vllm"),
        "outputs": {
            "parsed_docling_jsonl": OUT_PARSED_DOCLING_JSONL.as_posix(),
            "parsed_ocr_jsonl_compat": OUT_PARSED_OCR_JSONL.as_posix(),
            "parsed_docling_dir": PARSED_DOCLING_DIR.as_posix(),
            "docling_surya_failures_jsonl": FAILED_JSONL.as_posix(),
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
