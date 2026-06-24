#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mammoth
from docx import Document
from docx2python import docx2python
from tqdm import tqdm

from src.common import (
    CONVERTED_DIR,
    FAILED_DIR,
    OUTPUT_DIR,
    PARSED_WORD_DIR,
    LIBREOFFICE_CMD,
    WORD_EXTENSIONS,
    append_jsonl,
    clean_text,
    detect_language_from_text,
    ensure_base_dirs,
    get_quality_flags,
    iter_jsonl,
    read_jsonl,
    run_cmd,
    write_json,
    write_jsonl,
)


REGISTRY_JSONL = OUTPUT_DIR / "document_registry.jsonl"

OUT_PARSED_JSONL = OUTPUT_DIR / "parsed_word.jsonl"
OUT_SUMMARY_JSON = OUTPUT_DIR / "parsed_word_summary.json"

FAILED_JSONL = FAILED_DIR / "word_failures.jsonl"


# ----------------------------
# DOCX extractors
# ----------------------------

def extract_docx_mammoth(path: Path) -> str:
    with path.open("rb") as f:
        result = mammoth.extract_raw_text(f)
    return clean_text(result.value)


def extract_docx_python_docx(path: Path) -> str:
    doc = Document(str(path))
    blocks = []

    for p in doc.paragraphs:
        txt = clean_text(p.text)
        if txt:
            blocks.append(txt)

    for table in doc.tables:
        for row in table.rows:
            cells = [clean_text(cell.text) for cell in row.cells]
            cells = [c for c in cells if c]
            if cells:
                blocks.append(" | ".join(cells))

    return clean_text("\n".join(blocks))


def extract_docx_docx2python(path: Path) -> str:
    result = docx2python(str(path))
    try:
        text = result.text or ""
    finally:
        try:
            result.close()
        except Exception:
            pass

    return clean_text(text)


def extract_docx_best(path: Path) -> Tuple[str, str, Dict[str, int]]:
    candidates = []

    extractor_stats = {}

    for method_name, func in [
        ("docx_mammoth", extract_docx_mammoth),
        ("docx_python_docx", extract_docx_python_docx),
        ("docx_docx2python", extract_docx_docx2python),
    ]:
        try:
            text = func(path)
            extractor_stats[method_name] = len(text)
            candidates.append((method_name, text))
        except Exception:
            extractor_stats[method_name] = -1

    if not candidates:
        raise RuntimeError(f"All DOCX extractors failed: {path}")

    best_method, best_text = sorted(candidates, key=lambda x: len(x[1]), reverse=True)[0]
    return best_text, best_method, extractor_stats


# ----------------------------
# DOC → DOCX conversion
# ----------------------------

def libreoffice_convert_isolated(source_path: Path, target_ext: str, out_dir: Path, variant_id: str) -> Path:
    """
    Uses isolated LibreOffice profile to reduce parallel conversion conflicts.
    Useful when running multiple DOC conversions on server.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_dir = (CONVERTED_DIR / "lo_profiles" / variant_id).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_uri = profile_dir.as_uri()

    cmd = [
        LIBREOFFICE_CMD,
        f"-env:UserInstallation={profile_uri}",
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--convert-to",
        target_ext,
        "--outdir",
        str(out_dir),
        str(source_path),
    ]

    run_cmd(cmd, timeout=300)

    expected = out_dir / f"{source_path.stem}.{target_ext}"
    if expected.exists():
        return expected

    matches = list(out_dir.glob(f"{source_path.stem}*.{target_ext}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"LibreOffice output not found: {source_path} → {target_ext}")


def convert_doc_to_docx(source_path: Path, variant_id: str) -> Path:
    out_dir = CONVERTED_DIR / "doc_to_docx" / variant_id
    return libreoffice_convert_isolated(source_path, "docx", out_dir, variant_id)


# ----------------------------
# Worker
# ----------------------------

def extract_one_word_variant(row: Dict[str, Any], min_chars: int) -> Dict[str, Any]:
    source_path = Path(row["source_path"])
    source_format = row["source_format"]
    variant_id = row["variant_id"]

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    converted_path = None
    extraction_source_path = source_path

    if source_format == "doc":
        converted_path_obj = convert_doc_to_docx(source_path, variant_id)
        converted_path = converted_path_obj.as_posix()
        extraction_source_path = converted_path_obj

    elif source_format == "docx":
        extraction_source_path = source_path

    else:
        raise ValueError(f"Not a Word source_format: {source_format}")

    raw_text, extraction_method, extractor_stats = extract_docx_best(extraction_source_path)
    raw_text = clean_text(raw_text)

    detected_language = detect_language_from_text(raw_text)
    quality_flags = get_quality_flags(raw_text, detected_language)

    needs_ocr = False
    ocr_reason = None

    if len(raw_text) < min_chars:
        needs_ocr = True
        ocr_reason = "word_native_extraction_weak_low_char_count"
        if "low_char_count" not in quality_flags:
            quality_flags.append("low_char_count")

    record = {
        **row,
        "detected_language": detected_language,
        "extraction_method": extraction_method if source_format == "docx" else f"doc_to_docx_{extraction_method}",
        "converted_path": converted_path,
        "extractor_stats": extractor_stats,
        "char_count": len(raw_text),
        "quality_flags": quality_flags,
        "needs_ocr": needs_ocr,
        "ocr_reason": ocr_reason,
        "raw_text": raw_text,
    }

    return record


def safe_worker(args: Tuple[Dict[str, Any], int]) -> Tuple[str, Dict[str, Any]]:
    row, min_chars = args

    try:
        result = extract_one_word_variant(row, min_chars)
        return "ok", result

    except Exception as e:
        fail = {
            **row,
            "needs_ocr": True,
            "ocr_reason": "word_native_extraction_failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        return "failed", fail


# ----------------------------
# Load / save
# ----------------------------

def load_word_registry_rows(
    only_format: Optional[str] = None,
    only_unknown: bool = False,
) -> List[Dict[str, Any]]:
    rows = []

    for row in iter_jsonl(REGISTRY_JSONL):
        ext = row.get("source_extension")
        fmt = row.get("source_format")

        if ext not in WORD_EXTENSIONS:
            continue

        if only_format and fmt != only_format:
            continue

        if only_unknown and row.get("language_hint") != "unknown":
            continue

        rows.append(row)

    return rows


def rebuild_parsed_word_jsonl() -> int:
    rows = []

    for p in sorted(PARSED_WORD_DIR.glob("*.json")):
        try:
            rows.append(write_safe_read_json(p))
        except Exception:
            continue

    write_jsonl(OUT_PARSED_JSONL, rows)
    return len(rows)


def write_safe_read_json(path: Path) -> Dict[str, Any]:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--only-format", choices=["docx", "doc"], default=None)
    parser.add_argument("--only-unknown", action="store_true")
    args = parser.parse_args()

    ensure_base_dirs()

    rows = load_word_registry_rows(
        only_format=args.only_format,
        only_unknown=args.only_unknown,
    )

    rows = rows[args.start:]

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    if not rows:
        print("No Word registry rows found.")
        return

    print(f"Word rows selected: {len(rows)}")
    print(f"Workers: {args.workers}")
    print(f"Min chars: {args.min_chars}")

    tasks = []

    skipped_existing = 0

    for row in rows:
        out_path = PARSED_WORD_DIR / f'{row["variant_id"]}.json'
        if out_path.exists() and not args.force:
            skipped_existing += 1
            continue
        tasks.append(row)

    parsed_ok = 0
    parsed_weak_needs_ocr = 0
    failed = 0

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(safe_worker, (row, args.min_chars))
                for row in tasks
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Word"):
                status, result = future.result()

                if status == "ok":
                    out_path = PARSED_WORD_DIR / f'{result["variant_id"]}.json'
                    write_json(out_path, result)

                    parsed_ok += 1

                    if result.get("needs_ocr"):
                        parsed_weak_needs_ocr += 1
                        append_jsonl(FAILED_JSONL, {
                            "variant_id": result["variant_id"],
                            "source_path": result["source_path"],
                            "source_format": result["source_format"],
                            "language_hint": result.get("language_hint"),
                            "detected_language": result.get("detected_language"),
                            "needs_ocr": True,
                            "ocr_reason": result.get("ocr_reason"),
                            "char_count": result.get("char_count"),
                            "quality_flags": result.get("quality_flags", []),
                        })

                else:
                    failed += 1
                    append_jsonl(FAILED_JSONL, result)

    total_parsed_files = rebuild_parsed_word_jsonl()

    summary = {
        "registry_word_rows_considered": len(rows),
        "tasks_run": len(tasks),
        "skipped_existing": skipped_existing,
        "parsed_ok": parsed_ok,
        "parsed_weak_needs_ocr": parsed_weak_needs_ocr,
        "failed_needs_ocr": failed,
        "total_parsed_word_files": total_parsed_files,
        "outputs": {
            "parsed_word_jsonl": OUT_PARSED_JSONL.as_posix(),
            "parsed_word_dir": PARSED_WORD_DIR.as_posix(),
            "word_failures_jsonl": FAILED_JSONL.as_posix(),
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
