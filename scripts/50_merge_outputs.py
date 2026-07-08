#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    OUTPUT_DIR,
    PARSED_DOCLING_DIR,
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

# PDF-only merge. Word outputs are intentionally ignored in this pipeline.
PARSED_DOCLING_JSONL = OUTPUT_DIR / "parsed_docling.jsonl"

OUT_RAW_JSONL = OUTPUT_DIR / "parsed_guidelines_raw.jsonl"
OUT_GROUPED_JSON = OUTPUT_DIR / "parsed_guidelines_grouped_raw.json"
OUT_MISSING_CSV = OUTPUT_DIR / "missing_extracted_variants.csv"
OUT_SUMMARY_JSON = OUTPUT_DIR / "merge_outputs_summary.json"


def load_pdf_rows() -> List[Dict[str, Any]]:
    rows = list(iter_jsonl(PARSED_DOCLING_JSONL))

    if rows:
        return rows

    # Fallback if JSONL was not rebuilt.
    out = []
    for p in sorted(PARSED_DOCLING_DIR.glob("*/parsed.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


def normalize_pdf_record(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = clean_text(row.get("raw_text") or "")
    detected_language = row.get("detected_language") or detect_language_from_text(raw_text)
    final_language = detected_language if detected_language != "unknown" else row.get("language_hint", "unknown")

    quality_flags = list(row.get("quality_flags") or [])
    for flag in get_quality_flags(raw_text, detected_language):
        if flag not in quality_flags:
            quality_flags.append(flag)

    icd_codes = row.get("icd_codes_detected") or find_icd_codes(raw_text)

    # Ensure pending visual outputs are reflected at merge level.
    if int(row.get("pending_visual_count") or 0) > 0 and "pending_visual_assets" not in quality_flags:
        quality_flags.append("pending_visual_assets")

    if int(row.get("docling_batch_failed_count") or 0) > 0 and "docling_batch_errors" not in quality_flags:
        quality_flags.append("docling_batch_errors")

    return {
        "variant_id": row.get("variant_id"),
        "disease_class_id": row.get("disease_class_id"),
        "clinical_doc_group_id": row.get("clinical_doc_group_id"),
        "specialty": row.get("specialty"),
        "disease_path": row.get("disease_path"),
        "doc_type": row.get("doc_type"),

        "source_path": row.get("source_path") or row.get("original_source_path") or row.get("ocr_input_path"),
        "source_format": "pdf",
        "source_extension": ".pdf",
        "filename": row.get("filename"),
        "size_kb": row.get("size_kb"),

        "language_hint": row.get("language_hint", "unknown"),
        "detected_language": detected_language,
        "final_language": final_language,

        "extraction_source": "pdf_agentic_docling_surya",
        "extraction_method": row.get("extraction_method"),
        "char_count": len(raw_text),
        "quality_flags": sorted(set(quality_flags)),

        "markdown_path": row.get("markdown_path"),
        "agentic_json_path": row.get("agentic_json_path"),
        "raw_docling_dir": row.get("raw_docling_dir"),
        "visual_manifest_path": row.get("visual_manifest_path"),
        "visual_assets_dir": row.get("visual_assets_dir"),

        "pending_visual_count": int(row.get("pending_visual_count") or 0),
        "visual_asset_count": int(row.get("visual_asset_count") or 0),
        "docling_batch_count": int(row.get("docling_batch_count") or 0),
        "docling_batch_failed_count": int(row.get("docling_batch_failed_count") or 0),
        "page_count": int(row.get("page_count") or 0),

        "icd_codes_detected": icd_codes,
        "icd_code_count": len(icd_codes),

        "elapsed_seconds": row.get("elapsed_seconds"),
        "ocr_reason": row.get("ocr_reason"),
        "ocr_input_path": row.get("ocr_input_path"),
        "raw_text": raw_text,
    }


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
        key = (row.get("disease_class_id"), row.get("doc_type"))
        g = grouped[key]

        g["disease_class_id"] = row.get("disease_class_id")
        g["clinical_doc_group_id"] = row.get("clinical_doc_group_id")
        g["specialty"] = row.get("specialty")
        g["disease_path"] = row.get("disease_path")
        g["doc_type"] = row.get("doc_type")

        lang = row.get("final_language") or "unknown"
        g["available_languages"].add(lang)

        g["variants"][lang].append(
            {
                "variant_id": row.get("variant_id"),
                "source_path": row.get("source_path"),
                "source_format": row.get("source_format"),
                "language_hint": row.get("language_hint"),
                "detected_language": row.get("detected_language"),
                "final_language": row.get("final_language"),
                "extraction_source": row.get("extraction_source"),
                "extraction_method": row.get("extraction_method"),
                "char_count": row.get("char_count"),
                "quality_flags": row.get("quality_flags"),
                "icd_codes_detected": row.get("icd_codes_detected"),
                "markdown_path": row.get("markdown_path"),
                "agentic_json_path": row.get("agentic_json_path"),
                "raw_docling_dir": row.get("raw_docling_dir"),
                "visual_manifest_path": row.get("visual_manifest_path"),
                "pending_visual_count": row.get("pending_visual_count"),
            }
        )

    result = []
    for _, g in grouped.items():
        result.append(
            {
                "disease_class_id": g["disease_class_id"],
                "clinical_doc_group_id": g["clinical_doc_group_id"],
                "specialty": g["specialty"],
                "disease_path": g["disease_path"],
                "doc_type": g["doc_type"],
                "available_languages": sorted(g["available_languages"]),
                "variants": dict(g["variants"]),
            }
        )

    return sorted(result, key=lambda x: (str(x["specialty"]), str(x["disease_path"]), str(x["doc_type"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-chars", type=int, default=500)
    args = parser.parse_args()

    ensure_base_dirs()

    source_rows = load_pdf_rows()
    final_rows = [normalize_pdf_record(r) for r in source_rows]

    missing_rows = []
    for r in final_rows:
        if int(r.get("char_count") or 0) < args.min_chars:
            missing_rows.append({
                "variant_id": r.get("variant_id"),
                "source_path": r.get("source_path"),
                "reason": "char_count_below_min_after_pdf_extraction",
                "char_count": r.get("char_count"),
            })

    write_jsonl(OUT_RAW_JSONL, final_rows)
    write_json(OUT_GROUPED_JSON, build_grouped_output(final_rows))
    write_csv(OUT_MISSING_CSV, missing_rows)

    summary = {
        "mode": "pdf_only",
        "source_rows": len(source_rows),
        "final_rows": len(final_rows),
        "missing_or_low_quality_rows": len(missing_rows),
        "total_chars": sum(int(r.get("char_count") or 0) for r in final_rows),
        "total_pending_visual_assets": sum(int(r.get("pending_visual_count") or 0) for r in final_rows),
        "by_language": dict(Counter(r.get("final_language") or "unknown" for r in final_rows)),
        "by_quality_flag": dict(Counter(flag for r in final_rows for flag in (r.get("quality_flags") or []))),
    }
    write_json(OUT_SUMMARY_JSON, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
