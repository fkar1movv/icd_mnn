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
    PARSED_OCR_DIR,
    clean_text,
    detect_language_from_text,
    ensure_base_dirs,
    find_icd_codes,
    get_quality_flags,
    iter_jsonl,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)


REGISTRY_JSONL = OUTPUT_DIR / "document_registry.jsonl"
PARSED_WORD_JSONL = OUTPUT_DIR / "parsed_word.jsonl"
PARSED_OCR_JSONL = OUTPUT_DIR / "parsed_ocr.jsonl"

OUT_RAW_JSONL = OUTPUT_DIR / "parsed_guidelines_raw.jsonl"
OUT_GROUPED_JSON = OUTPUT_DIR / "parsed_guidelines_grouped_raw.json"
OUT_MISSING_CSV = OUTPUT_DIR / "missing_extracted_variants.csv"
OUT_SUMMARY_JSON = OUTPUT_DIR / "merge_outputs_summary.json"


def load_records_by_variant(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}

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


def load_word_fallback_from_dir() -> Dict[str, Dict[str, Any]]:
    rows = {}

    for p in sorted(PARSED_WORD_DIR.glob("*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            variant_id = row.get("variant_id")
            if variant_id:
                rows[variant_id] = row
        except Exception:
            pass

    return rows


def load_ocr_fallback_from_dir() -> Dict[str, Dict[str, Any]]:
    rows = {}

    for p in sorted(PARSED_OCR_DIR.glob("*/parsed.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            variant_id = row.get("variant_id")
            if variant_id:
                rows[variant_id] = row
        except Exception:
            pass

    return rows


def final_language(record: Dict[str, Any]) -> str:
    detected = record.get("detected_language") or "unknown"
    hint = record.get("language_hint") or "unknown"

    if detected != "unknown":
        return detected

    if hint != "unknown":
        return hint

    return "unknown"


def normalize_final_record(
    registry_row: Dict[str, Any],
    extracted_row: Dict[str, Any],
    extraction_source: str,
) -> Dict[str, Any]:
    raw_text = clean_text(extracted_row.get("raw_text") or extracted_row.get("text") or "")

    detected_language = extracted_row.get("detected_language") or detect_language_from_text(raw_text)
    lang_final = detected_language if detected_language != "unknown" else registry_row.get("language_hint", "unknown")

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

        # Extraction metadata
        "extraction_source": extraction_source,  # word / ocr
        "extraction_method": extracted_row.get("extraction_method"),
        "converted_path": extracted_row.get("converted_path"),
        "raw_output_dir": extracted_row.get("raw_output_dir"),
        "char_count": len(raw_text),
        "quality_flags": quality_flags,

        # Deterministic prechecks
        "icd_codes_detected": icd_codes,
        "icd_code_count": len(icd_codes),

        # Important: raw source text is preserved.
        "raw_text": raw_text,
    }

    # Keep OCR-specific fields if present.
    for k in [
        "ocr_reason",
        "ocr_input_path",
        "original_source_path",
        "from_word_conversion",
        "elapsed_seconds",
        "saved_outputs",
    ]:
        if k in extracted_row:
            record[k] = extracted_row.get(k)

    # Keep Word extractor stats if present.
    if "extractor_stats" in extracted_row:
        record["extractor_stats"] = extracted_row.get("extractor_stats")

    return record


def select_extraction(
    registry_row: Dict[str, Any],
    word_by_variant: Dict[str, Dict[str, Any]],
    ocr_by_variant: Dict[str, Dict[str, Any]],
    min_chars: int,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    """
    Priority:
      PDF registry row:
        OCR required.

      Word registry row:
        Good native Word extraction preferred.
        If Word was weak/failed and OCR exists, use OCR.
        If Word weak and OCR missing, keep weak Word but mark quality.
    """

    variant_id = registry_row["variant_id"]
    source_format = registry_row.get("source_format")

    word = word_by_variant.get(variant_id)
    ocr = ocr_by_variant.get(variant_id)

    if source_format == "pdf":
        if ocr:
            return "ocr", ocr, "pdf_ocr_available"
        return None, None, "missing_pdf_ocr"

    if source_format in {"docx", "doc"}:
        if word:
            word_chars = int(word.get("char_count") or len(word.get("raw_text") or ""))
            word_needs_ocr = bool(word.get("needs_ocr"))

            if not word_needs_ocr and word_chars >= min_chars:
                return "word", word, "word_native_good"

            if ocr:
                return "ocr", ocr, "word_native_weak_or_failed_used_ocr"

            # Keep weak Word extraction if OCR is not available yet.
            if word:
                qf = list(word.get("quality_flags") or [])
                if "used_weak_word_no_ocr_available" not in qf:
                    qf.append("used_weak_word_no_ocr_available")
                word["quality_flags"] = qf
                return "word", word, "word_weak_no_ocr_available"

        if ocr:
            return "ocr", ocr, "word_failed_used_ocr"

        return None, None, "missing_word_extraction"

    return None, None, f"unsupported_source_format_{source_format}"


def build_grouped_output(final_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(lambda: {
        "disease_class_id": None,
        "clinical_doc_group_id": None,
        "specialty": None,
        "disease_path": None,
        "doc_type": None,
        "available_languages": set(),
        "variants": defaultdict(list),
    })

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

        # Keep metadata only in grouped file, not full raw_text duplication.
        g["variants"][lang].append({
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
        })

    result = []

    for _, g in grouped.items():
        variants = {lang: rows for lang, rows in g["variants"].items()}
        result.append({
            "disease_class_id": g["disease_class_id"],
            "clinical_doc_group_id": g["clinical_doc_group_id"],
            "specialty": g["specialty"],
            "disease_path": g["disease_path"],
            "doc_type": g["doc_type"],
            "available_languages": sorted(g["available_languages"]),
            "variants": variants,
        })

    return sorted(result, key=lambda x: (x["specialty"], x["disease_path"], x["doc_type"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-chars", type=int, default=500)
    args = parser.parse_args()

    ensure_base_dirs()

    registry_rows = list(iter_jsonl(REGISTRY_JSONL))
    if not registry_rows:
        raise FileNotFoundError(f"No registry rows found: {REGISTRY_JSONL}")

    word_by_variant = load_records_by_variant(PARSED_WORD_JSONL)
    if not word_by_variant:
        word_by_variant = load_word_fallback_from_dir()

    ocr_by_variant = load_records_by_variant(PARSED_OCR_JSONL)
    if not ocr_by_variant:
        ocr_by_variant = load_ocr_fallback_from_dir()

    final_rows = []
    missing_rows = []
    selection_reasons = Counter()

    for reg in registry_rows:
        extraction_source, extracted, reason = select_extraction(
            reg,
            word_by_variant,
            ocr_by_variant,
            min_chars=args.min_chars,
        )

        selection_reasons[reason] += 1

        if not extracted or not extraction_source:
            missing_rows.append({
                "variant_id": reg.get("variant_id"),
                "source_path": reg.get("source_path"),
                "source_format": reg.get("source_format"),
                "language_hint": reg.get("language_hint"),
                "specialty": reg.get("specialty"),
                "disease_path": reg.get("disease_path"),
                "doc_type": reg.get("doc_type"),
                "missing_reason": reason,
            })
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
        )
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
        "ocr_records_loaded": len(ocr_by_variant),
        "final_raw_records": len(final_rows),
        "missing_records": len(missing_rows),
        "clinical_doc_groups": len(grouped),
        "by_extraction_source": dict(Counter(x["extraction_source"] for x in final_rows)),
        "by_source_format": dict(Counter(x["source_format"] for x in final_rows)),
        "by_final_language": dict(Counter(x["final_language"] for x in final_rows)),
        "by_doc_type": dict(Counter(x["doc_type"] for x in final_rows)),
        "quality_flag_counts": dict(Counter(flag for x in final_rows for flag in x.get("quality_flags", []))),
        "selection_reasons": dict(selection_reasons),
        "outputs": {
            "parsed_guidelines_raw_jsonl": OUT_RAW_JSONL.as_posix(),
            "parsed_guidelines_grouped_raw_json": OUT_GROUPED_JSON.as_posix(),
            "missing_extracted_variants_csv": OUT_MISSING_CSV.as_posix(),
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
