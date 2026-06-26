#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    OUTPUT_DIR,
    PARSED_WORD_DIR,
    PARSED_DOCLING_DIR,
    PARSED_OCR_DIR,
    clean_text,
    detect_language_from_text,
    ensure_base_dirs,
    find_icd_codes,
    get_quality_flags,
    iter_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)


REGISTRY_JSONL = OUTPUT_DIR / "document_registry.jsonl"

PARSED_WORD_JSONL = OUTPUT_DIR / "parsed_word.jsonl"

# New primary PDF extraction output.
PARSED_DOCLING_JSONL = OUTPUT_DIR / "parsed_docling.jsonl"

# Compatibility alias. scripts/40_run_docling_surya.py also writes this.
PARSED_OCR_JSONL = OUTPUT_DIR / "parsed_ocr.jsonl"

OUT_RAW_JSONL = OUTPUT_DIR / "parsed_guidelines_raw.jsonl"
OUT_GROUPED_JSON = OUTPUT_DIR / "parsed_guidelines_grouped_raw.json"
OUT_MISSING_CSV = OUTPUT_DIR / "missing_extracted_variants.csv"
OUT_SUMMARY_JSON = OUTPUT_DIR / "merge_outputs_summary.json"


# ============================================================
# Load helpers
# ============================================================

def load_records_by_variant(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}

    if not path.exists():
        return rows

    for row in iter_jsonl(path):
        variant_id = row.get("variant_id")

        if not variant_id:
            continue

        # If duplicate exists, keep longer extraction.
        old = rows.get(variant_id)

        if old is None:
            rows[variant_id] = row
        else:
            if int(row.get("char_count") or 0) > int(old.get("char_count") or 0):
                rows[variant_id] = row

    return rows


def merge_record_maps(
    primary: Dict[str, Dict[str, Any]],
    fallback: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Merge by variant_id. If same variant exists in both maps, keep the longer one.
    """

    result = dict(primary)

    for variant_id, row in fallback.items():
        old = result.get(variant_id)

        if old is None:
            result[variant_id] = row
            continue

        if int(row.get("char_count") or 0) > int(old.get("char_count") or 0):
            result[variant_id] = row

    return result


def load_word_fallback_from_dir() -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}

    for p in sorted(PARSED_WORD_DIR.glob("*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            variant_id = row.get("variant_id")

            if variant_id:
                rows[variant_id] = row

        except Exception:
            pass

    return rows


def load_docling_fallback_from_dir() -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}

    for p in sorted(PARSED_DOCLING_DIR.glob("*/parsed.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            variant_id = row.get("variant_id")

            if variant_id:
                rows[variant_id] = row

        except Exception:
            pass

    return rows


def load_ocr_compat_fallback_from_dir() -> Dict[str, Dict[str, Any]]:
    """
    Compatibility fallback only.

    In the new pipeline PARSED_OCR_DIR should be the same path as
    PARSED_DOCLING_DIR, but this keeps old directory layouts from breaking.
    """

    rows: Dict[str, Dict[str, Any]] = {}

    if PARSED_OCR_DIR == PARSED_DOCLING_DIR:
        return rows

    for p in sorted(PARSED_OCR_DIR.glob("*/parsed.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            variant_id = row.get("variant_id")

            if variant_id:
                rows[variant_id] = row

        except Exception:
            pass

    return rows


# ============================================================
# Final normalization
# ============================================================

def normalize_final_record(
    registry_row: Dict[str, Any],
    extracted_row: Dict[str, Any],
    extraction_source: str,
) -> Dict[str, Any]:
    raw_text = clean_text(extracted_row.get("raw_text") or extracted_row.get("text") or "")

    detected_language = extracted_row.get("detected_language") or detect_language_from_text(raw_text)

    if detected_language != "unknown":
        lang_final = detected_language
    else:
        lang_final = registry_row.get("language_hint", "unknown")

    quality_flags = list(extracted_row.get("quality_flags") or [])

    for flag in get_quality_flags(raw_text, detected_language):
        if flag not in quality_flags:
            quality_flags.append(flag)

    icd_codes = find_icd_codes(raw_text)

    record = {
        # Registry identity
        "variant_id": registry_row.get("variant_id"),
        "disease_class_id": registry_row.get("disease_class_id"),
        "clinical_doc_group_id": registry_row.get("clinical_doc_group_id"),
        "specialty": registry_row.get("specialty"),
        "disease_path": registry_row.get("disease_path"),
        "doc_type": registry_row.get("doc_type"),

        # Source identity
        "source_path": registry_row.get("source_path"),
        "source_format": registry_row.get("source_format"),
        "source_extension": registry_row.get("source_extension"),
        "filename": registry_row.get("filename"),
        "size_kb": registry_row.get("size_kb"),

        # Language
        "language_hint": registry_row.get("language_hint", "unknown"),
        "detected_language": detected_language,
        "final_language": lang_final,

        # Final extraction type
        # Values:
        #   word    = DOC/DOCX native Python + Docling Word pipeline
        #   docling = PDF Docling + SuryaOCR pipeline
        "extraction_source": extraction_source,
        "extraction_method": extracted_row.get("extraction_method"),
        "char_count": len(raw_text),
        "quality_flags": quality_flags,

        # Paths to generated artifacts
        "markdown_path": extracted_row.get("markdown_path"),
        "docling_json_path": extracted_row.get("docling_json_path"),

        # Deterministic prechecks
        "icd_codes_detected": icd_codes,
        "icd_code_count": len(icd_codes),

        # Preserve source text for later local cleanup/mapping.
        "raw_text": raw_text,
    }

    # Keep conversion/extraction details when present.
    passthrough_keys = [
        "converted_path",
        "raw_output_dir",
        "elapsed_seconds",
        "docling_counts",
        "ocr_reason",
        "ocr_input_path",
        "original_source_path",
        "from_word_conversion",
        "saved_outputs",
        "extractor_stats",
        "native_method",
        "native_char_count",
        "docling_char_count",
        "native_error",
        "docling_error",
        "needs_review",
        "review_reason",
        "word_extraction_source",
    ]

    for k in passthrough_keys:
        if k in extracted_row:
            record[k] = extracted_row.get(k)

    # Keep the internal Word extraction source separately.
    if extraction_source == "word":
        record["word_extraction_source"] = extracted_row.get("extraction_source")

    return record


# ============================================================
# Selection
# ============================================================

def select_extraction(
    registry_row: Dict[str, Any],
    word_by_variant: Dict[str, Dict[str, Any]],
    docling_by_variant: Dict[str, Dict[str, Any]],
    min_chars: int,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    """
    Priority:

      PDF registry row:
        Use Docling + SuryaOCR result.

      Word registry row:
        Use Word result from native Python + Docling Word extraction.
        We do not convert weak Word to PDF OCR in the new pipeline.
    """

    variant_id = registry_row["variant_id"]
    source_format = registry_row.get("source_format")

    word = word_by_variant.get(variant_id)
    docling = docling_by_variant.get(variant_id)

    if source_format == "pdf":
        if docling:
            return "docling", docling, "pdf_docling_surya_available"

        return None, None, "missing_pdf_docling_surya"

    if source_format in {"docx", "doc"}:
        if word:
            word_chars = int(word.get("char_count") or len(word.get("raw_text") or ""))

            if word_chars >= min_chars:
                return "word", word, "word_extraction_available"

            # Keep weak Word extraction if it exists; mark it for review.
            qf = list(word.get("quality_flags") or [])

            if "used_weak_word_extraction" not in qf:
                qf.append("used_weak_word_extraction")

            word["quality_flags"] = qf
            word["needs_review"] = True
            word["review_reason"] = word.get("review_reason") or "word_extraction_below_min_chars"

            return "word", word, "word_weak_but_available"

        return None, None, "missing_word_extraction"

    return None, None, f"unsupported_source_format_{source_format}"


# ============================================================
# Grouped output
# ============================================================

def build_grouped_output(final_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(
        lambda: {
            "disease_class_id": None,
            "clinical_doc_group_id": None,
            "specialty": None,
            "disease_path": None,
            "doc_type": None,
            "available_languages": set(),
            "variants": defaultdict(list),
        }
    )

    for row in final_rows:
        key = (row["disease_class_id"], row["doc_type"])
        g = grouped[key]

        g["disease_class_id"] = row["disease_class_id"]
        g["clinical_doc_group_id"] = row["clinical_doc_group_id"]
        g["specialty"] = row["specialty"]
        g["disease_path"] = row["disease_path"]
        g["doc_type"] = row["doc_type"]

        lang = row.get("final_language") or "unknown"
        g["available_languages"].add(lang)

        # Keep metadata only. Do not duplicate raw_text in grouped JSON.
        g["variants"][lang].append(
            {
                "variant_id": row["variant_id"],
                "source_path": row["source_path"],
                "source_format": row["source_format"],
                "language_hint": row["language_hint"],
                "detected_language": row["detected_language"],
                "final_language": row["final_language"],
                "extraction_source": row["extraction_source"],
                "extraction_method": row["extraction_method"],
                "char_count": row["char_count"],
                "quality_flags": row["quality_flags"],
                "icd_codes_detected": row["icd_codes_detected"],
                "markdown_path": row.get("markdown_path"),
                "docling_json_path": row.get("docling_json_path"),
            }
        )

    result = []

    for _, g in grouped.items():
        variants = {lang: rows for lang, rows in g["variants"].items()}

        result.append(
            {
                "disease_class_id": g["disease_class_id"],
                "clinical_doc_group_id": g["clinical_doc_group_id"],
                "specialty": g["specialty"],
                "disease_path": g["disease_path"],
                "doc_type": g["doc_type"],
                "available_languages": sorted(g["available_languages"]),
                "variants": variants,
            }
        )

    return sorted(
        result,
        key=lambda x: (
            x["specialty"],
            x["disease_path"],
            x["doc_type"],
        ),
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-chars", type=int, default=500)
    args = parser.parse_args()

    ensure_base_dirs()

    registry_rows = list(iter_jsonl(REGISTRY_JSONL))

    if not registry_rows:
        raise FileNotFoundError(f"No registry rows found: {REGISTRY_JSONL}")

    # Word records.
    word_by_variant = load_records_by_variant(PARSED_WORD_JSONL)
    word_by_variant = merge_record_maps(
        word_by_variant,
        load_word_fallback_from_dir(),
    )

    # Docling PDF records.
    docling_by_variant = load_records_by_variant(PARSED_DOCLING_JSONL)

    # Compatibility fallback: old or alias parsed_ocr.jsonl.
    docling_by_variant = merge_record_maps(
        docling_by_variant,
        load_records_by_variant(PARSED_OCR_JSONL),
    )

    # Directory fallback.
    docling_by_variant = merge_record_maps(
        docling_by_variant,
        load_docling_fallback_from_dir(),
    )

    docling_by_variant = merge_record_maps(
        docling_by_variant,
        load_ocr_compat_fallback_from_dir(),
    )

    final_rows = []
    missing_rows = []
    selection_reasons = Counter()

    for reg in registry_rows:
        extraction_source, extracted, reason = select_extraction(
            reg,
            word_by_variant,
            docling_by_variant,
            min_chars=args.min_chars,
        )

        selection_reasons[reason] += 1

        if not extracted or not extraction_source:
            missing_rows.append(
                {
                    "variant_id": reg.get("variant_id"),
                    "source_path": reg.get("source_path"),
                    "source_format": reg.get("source_format"),
                    "language_hint": reg.get("language_hint"),
                    "specialty": reg.get("specialty"),
                    "disease_path": reg.get("disease_path"),
                    "doc_type": reg.get("doc_type"),
                    "missing_reason": reason,
                }
            )
            continue

        final_record = normalize_final_record(reg, extracted, extraction_source)
        final_record["merge_selection_reason"] = reason
        final_rows.append(final_record)

    final_rows = sorted(
        final_rows,
        key=lambda x: (
            x.get("specialty") or "",
            x.get("disease_path") or "",
            x.get("doc_type") or "",
            x.get("final_language") or "",
            x.get("source_path") or "",
        ),
    )

    grouped = build_grouped_output(final_rows)

    missing_fields = [
        "variant_id",
        "source_path",
        "source_format",
        "language_hint",
        "specialty",
        "disease_path",
        "doc_type",
        "missing_reason",
    ]

    write_jsonl(OUT_RAW_JSONL, final_rows)
    write_json(OUT_GROUPED_JSON, grouped)
    write_csv(OUT_MISSING_CSV, missing_rows, missing_fields)

    summary = {
        "registry_total": len(registry_rows),
        "word_records_loaded": len(word_by_variant),
        "docling_records_loaded": len(docling_by_variant),
        "final_raw_records": len(final_rows),
        "missing_records": len(missing_rows),
        "clinical_doc_groups": len(grouped),
        "by_extraction_source": dict(Counter(x["extraction_source"] for x in final_rows)),
        "by_source_format": dict(Counter(x["source_format"] for x in final_rows)),
        "by_final_language": dict(Counter(x["final_language"] for x in final_rows)),
        "by_doc_type": dict(Counter(x["doc_type"] for x in final_rows)),
        "quality_flag_counts": dict(
            Counter(flag for x in final_rows for flag in x.get("quality_flags", []))
        ),
        "selection_reasons": dict(selection_reasons),
        "outputs": {
            "parsed_guidelines_raw_jsonl": OUT_RAW_JSONL.as_posix(),
            "parsed_guidelines_grouped_raw_json": OUT_GROUPED_JSON.as_posix(),
            "missing_extracted_variants_csv": OUT_MISSING_CSV.as_posix(),
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
