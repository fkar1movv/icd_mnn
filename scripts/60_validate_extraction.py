#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    OUTPUT_DIR,
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

RAW_JSONL = OUTPUT_DIR / "parsed_guidelines_raw.jsonl"
CLEANED_JSONL = OUTPUT_DIR / "parsed_guidelines_cleaned.jsonl"

OUT_REPORT_JSON = OUTPUT_DIR / "extraction_quality_report.json"
OUT_REPORT_CSV = OUTPUT_DIR / "extraction_quality_report.csv"
OUT_NEEDS_REVIEW_CSV = OUTPUT_DIR / "needs_review.csv"
OUT_VISUAL_REVIEW_CSV = OUTPUT_DIR / "pending_visual_assets_review.csv"
OUT_DUPLICATES_CSV = OUTPUT_DIR / "possible_duplicate_texts.csv"


BAD_TEXT_PATTERNS = [
    r"�{3,}",
    r"(.)\1{20,}",
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
]


def load_source_rows(prefer_cleaned: bool = False) -> List[Dict[str, Any]]:
    source = CLEANED_JSONL if prefer_cleaned and CLEANED_JSONL.exists() else RAW_JSONL
    rows = list(iter_jsonl(source))

    for row in rows:
        if prefer_cleaned and row.get("clean_text"):
            row["_validation_text"] = row.get("clean_text") or ""
            row["_validation_text_source"] = "clean_text"
        else:
            row["_validation_text"] = row.get("raw_text") or ""
            row["_validation_text_source"] = "raw_text"

    return rows


def path_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()


def text_hash(text: str) -> str:
    import hashlib

    normalized = clean_text(text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.md5(normalized[:50000].encode("utf-8")).hexdigest()


def count_bad_patterns(text: str) -> int:
    count = 0
    for p in BAD_TEXT_PATTERNS:
        count += len(re.findall(p, text))
    return count


def ratio_count(pattern: str, text: str) -> float:
    if not text:
        return 0.0
    return round(len(re.findall(pattern, text)) / max(len(text), 1), 4)


def calc_letter_ratio(text: str) -> float:
    return ratio_count(r"[A-Za-zА-Яа-яЁёҚқЎўҒғҲҳ]", text)


def calc_digit_ratio(text: str) -> float:
    return ratio_count(r"\d", text)


def calc_cyrillic_ratio(text: str) -> float:
    return ratio_count(r"[А-Яа-яЁёҚқЎўҒғҲҳ]", text)


def calc_latin_ratio(text: str) -> float:
    return ratio_count(r"[A-Za-z]", text)


def read_manifest_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path or "")
    if not p.exists():
        return []
    return list(iter_jsonl(p))


def detect_review_flags(row: Dict[str, Any], min_chars: int) -> List[str]:
    text = clean_text(row.get("_validation_text") or "")
    detected_language = row.get("detected_language") or detect_language_from_text(text)
    flags = list(row.get("quality_flags") or [])

    for f in get_quality_flags(text, detected_language):
        if f not in flags:
            flags.append(f)

    char_count = len(text)
    icd_codes = find_icd_codes(text)

    if row.get("source_format") != "pdf":
        flags.append("non_pdf_row_in_pdf_only_validation")

    if char_count < min_chars:
        flags.append("needs_review_char_count_below_min")

    if not icd_codes:
        flags.append("needs_review_no_icd_codes_detected")

    if count_bad_patterns(text) > 0:
        flags.append("needs_review_bad_text_patterns")

    if char_count > 500 and calc_letter_ratio(text) < 0.25:
        flags.append("needs_review_low_letter_ratio")

    if row.get("final_language") == "unknown" or detected_language == "unknown":
        flags.append("needs_review_unknown_language")

    # Required extraction artifacts.
    if not path_exists(row.get("markdown_path")):
        flags.append("missing_markdown_output")

    if not path_exists(row.get("agentic_json_path")):
        flags.append("missing_agentic_json_output")

    if not path_exists(row.get("raw_docling_dir")):
        flags.append("missing_raw_docling_dir")

    # Visual assets.
    pending_count = int(row.get("pending_visual_count") or 0)
    manifest_path = row.get("visual_manifest_path")

    if pending_count > 0:
        flags.append("pending_visual_assets")

        if not path_exists(manifest_path):
            flags.append("missing_visual_manifest")
        else:
            manifest_rows = read_manifest_rows(manifest_path)

            if len(manifest_rows) != pending_count:
                flags.append("visual_manifest_count_mismatch")

            missing_crops = [m for m in manifest_rows if not path_exists(m.get("crop_path"))]
            if missing_crops:
                flags.append("missing_visual_crop_files")

    if int(row.get("docling_batch_failed_count") or 0) > 0:
        flags.append("docling_batch_failures")

    if int(row.get("page_count") or 0) <= 0:
        flags.append("missing_page_count")

    # Deduplicate preserving order.
    out = []
    seen = set()
    for f in flags:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def build_quality_row(row: Dict[str, Any], min_chars: int) -> Dict[str, Any]:
    text = clean_text(row.get("_validation_text") or "")
    detected_language = row.get("detected_language") or detect_language_from_text(text)
    final_language = row.get("final_language") or detected_language or row.get("language_hint") or "unknown"
    icd_codes = find_icd_codes(text)
    review_flags = detect_review_flags(row, min_chars)

    return {
        "variant_id": row.get("variant_id"),
        "disease_class_id": row.get("disease_class_id"),
        "clinical_doc_group_id": row.get("clinical_doc_group_id"),
        "specialty": row.get("specialty"),
        "disease_path": row.get("disease_path"),
        "doc_type": row.get("doc_type"),
        "source_path": row.get("source_path"),
        "source_format": row.get("source_format"),
        "extraction_source": row.get("extraction_source"),
        "extraction_method": row.get("extraction_method"),
        "language_hint": row.get("language_hint"),
        "detected_language": detected_language,
        "final_language": final_language,
        "validation_text_source": row.get("_validation_text_source"),
        "char_count": len(text),
        "page_count": row.get("page_count"),
        "letter_ratio": calc_letter_ratio(text),
        "digit_ratio": calc_digit_ratio(text),
        "cyrillic_ratio": calc_cyrillic_ratio(text),
        "latin_ratio": calc_latin_ratio(text),
        "icd_code_count": len(icd_codes),
        "icd_codes_detected": ", ".join(icd_codes[:50]),
        "bad_pattern_count": count_bad_patterns(text),
        "pending_visual_count": row.get("pending_visual_count"),
        "visual_asset_count": row.get("visual_asset_count"),
        "docling_batch_count": row.get("docling_batch_count"),
        "docling_batch_failed_count": row.get("docling_batch_failed_count"),
        "markdown_path": row.get("markdown_path"),
        "agentic_json_path": row.get("agentic_json_path"),
        "raw_docling_dir": row.get("raw_docling_dir"),
        "visual_manifest_path": row.get("visual_manifest_path"),
        "review_flag_count": len(review_flags),
        "review_flags": "; ".join(review_flags),
        "needs_review": bool(review_flags),
    }


def collect_visual_review_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for row in rows:
        manifest_path = row.get("visual_manifest_path")
        for item in read_manifest_rows(manifest_path):
            out.append({
                "variant_id": row.get("variant_id"),
                "source_path": row.get("source_path"),
                "asset_id": item.get("asset_id"),
                "page": item.get("page"),
                "order": item.get("order"),
                "suggested_type": item.get("suggested_type"),
                "crop_path": item.get("crop_path"),
                "crop_exists": path_exists(item.get("crop_path")),
                "status": item.get("status"),
                "caption": item.get("caption"),
            })

    return out


def find_duplicates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets = defaultdict(list)

    for row in rows:
        text = row.get("_validation_text") or ""
        if len(text) < 1000:
            continue
        buckets[text_hash(text)].append(row)

    duplicate_rows = []

    for h, items in buckets.items():
        if len(items) <= 1:
            continue

        for row in items:
            duplicate_rows.append({
                "text_hash": h,
                "duplicate_group_size": len(items),
                "variant_id": row.get("variant_id"),
                "source_path": row.get("source_path"),
                "specialty": row.get("specialty"),
                "disease_path": row.get("disease_path"),
                "doc_type": row.get("doc_type"),
                "final_language": row.get("final_language"),
                "char_count": row.get("char_count"),
            })

    return duplicate_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefer-cleaned", action="store_true")
    parser.add_argument("--min-chars", type=int, default=500)
    args = parser.parse_args()

    ensure_base_dirs()

    rows = load_source_rows(prefer_cleaned=args.prefer_cleaned)

    if not rows:
        raise FileNotFoundError(f"No extraction rows found: {RAW_JSONL}")

    quality_rows = [build_quality_row(row, min_chars=args.min_chars) for row in rows]
    needs_review_rows = [r for r in quality_rows if r["needs_review"]]
    duplicate_rows = find_duplicates(rows)
    visual_review_rows = collect_visual_review_rows(rows)

    flag_counter = Counter()
    for row in quality_rows:
        for f in str(row.get("review_flags") or "").split("; "):
            if f:
                flag_counter[f] += 1

    report = {
        "mode": "pdf_only_agentic_docling_surya",
        "total_rows": len(quality_rows),
        "needs_review_total": len(needs_review_rows),
        "visual_asset_total": len(visual_review_rows),
        "total_chars": sum(int(r.get("char_count") or 0) for r in quality_rows),
        "by_source_format": dict(Counter(r.get("source_format") for r in quality_rows)),
        "by_extraction_source": dict(Counter(r.get("extraction_source") for r in quality_rows)),
        "by_detected_language": dict(Counter(r.get("detected_language") for r in quality_rows)),
        "by_final_language": dict(Counter(r.get("final_language") for r in quality_rows)),
        "review_flags": dict(flag_counter.most_common()),
        "duplicate_groups": len(set(r["text_hash"] for r in duplicate_rows)),
    }

    write_json(OUT_REPORT_JSON, report)
    write_csv(OUT_REPORT_CSV, quality_rows)
    write_csv(OUT_NEEDS_REVIEW_CSV, needs_review_rows)
    write_csv(OUT_VISUAL_REVIEW_CSV, visual_review_rows)
    write_csv(OUT_DUPLICATES_CSV, duplicate_rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
