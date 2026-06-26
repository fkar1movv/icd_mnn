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
    REQUEST_LANGUAGES,
    FALLBACK_LANGUAGE_PRIORITY,
    clean_text,
    detect_language_from_text,
    ensure_base_dirs,
    find_icd_codes,
    get_quality_flags,
    iter_jsonl,
    write_csv,
    write_json,
)


RAW_JSONL = OUTPUT_DIR / "parsed_guidelines_raw.jsonl"
CLEANED_JSONL = OUTPUT_DIR / "parsed_guidelines_cleaned.jsonl"

OUT_REPORT_JSON = OUTPUT_DIR / "extraction_quality_report.json"
OUT_REPORT_CSV = OUTPUT_DIR / "extraction_quality_report.csv"
OUT_NEEDS_REVIEW_CSV = OUTPUT_DIR / "needs_review.csv"
OUT_LANGUAGE_FALLBACK_CSV = OUTPUT_DIR / "language_fallback_after_extraction.csv"
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


def calc_letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = len(re.findall(r"[A-Za-zА-Яа-яЁёҚқЎўҒғҲҳ]", text))
    return round(letters / max(len(text), 1), 4)


def calc_digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = len(re.findall(r"\d", text))
    return round(digits / max(len(text), 1), 4)


def calc_cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    chars = len(re.findall(r"[А-Яа-яЁёҚқЎўҒғҲҳ]", text))
    return round(chars / max(len(text), 1), 4)


def calc_latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    chars = len(re.findall(r"[A-Za-z]", text))
    return round(chars / max(len(text), 1), 4)


def detect_review_flags(row: Dict[str, Any], min_chars: int) -> List[str]:
    text = clean_text(row.get("_validation_text") or "")
    detected_language = row.get("detected_language") or detect_language_from_text(text)
    flags = list(row.get("quality_flags") or [])

    for f in get_quality_flags(text, detected_language):
        if f not in flags:
            flags.append(f)

    char_count = len(text)
    icd_codes = find_icd_codes(text)

    if char_count < min_chars:
        flags.append("needs_review_char_count_below_min")

    if not icd_codes:
        flags.append("needs_review_no_icd_codes_detected")

    if count_bad_patterns(text) > 0:
        flags.append("needs_review_bad_text_patterns")

    letter_ratio = calc_letter_ratio(text)
    if char_count > 500 and letter_ratio < 0.25:
        flags.append("needs_review_low_letter_ratio")

    if row.get("final_language") == "unknown" or detected_language == "unknown":
        flags.append("needs_review_unknown_language")

    if row.get("extraction_source") == "ocr" and char_count < 1000:
        flags.append("needs_review_short_ocr_output")

    if row.get("source_format") == "pdf" and row.get("extraction_source") != "ocr":
        flags.append("needs_review_pdf_without_ocr")

    # Deduplicate flags preserving order.
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
        "letter_ratio": calc_letter_ratio(text),
        "digit_ratio": calc_digit_ratio(text),
        "cyrillic_ratio": calc_cyrillic_ratio(text),
        "latin_ratio": calc_latin_ratio(text),
        "icd_code_count": len(icd_codes),
        "icd_codes_detected": ", ".join(icd_codes[:50]),
        "bad_pattern_count": count_bad_patterns(text),
        "review_flag_count": len(review_flags),
        "review_flags": "; ".join(review_flags),
        "needs_review": bool(review_flags),
    }


def build_fallback_after_extraction(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(lambda: {
        "disease_class_id": None,
        "clinical_doc_group_id": None,
        "specialty": None,
        "disease_path": None,
        "doc_type": None,
        "variants": defaultdict(list),
    })

    for row in rows:
        lang = row.get("final_language") or row.get("detected_language") or "unknown"
        key = (row.get("disease_class_id"), row.get("doc_type"))

        g = grouped[key]
        g["disease_class_id"] = row.get("disease_class_id")
        g["clinical_doc_group_id"] = row.get("clinical_doc_group_id")
        g["specialty"] = row.get("specialty")
        g["disease_path"] = row.get("disease_path")
        g["doc_type"] = row.get("doc_type")
        g["variants"][lang].append(row)

    fallback_rows = []

    for _, group in grouped.items():
        variants = group["variants"]

        for requested_lang in REQUEST_LANGUAGES:
            resolved_lang = None
            resolved_variant = None
            fallback_type = None

            if requested_lang in variants:
                resolved_lang = requested_lang
                resolved_variant = sorted(
                    variants[requested_lang],
                    key=lambda x: int(x.get("char_count") or 0),
                    reverse=True,
                )[0]
                fallback_type = "exact"
            else:
                for lang in FALLBACK_LANGUAGE_PRIORITY:
                    if lang in variants:
                        resolved_lang = lang
                        resolved_variant = sorted(
                            variants[lang],
                            key=lambda x: int(x.get("char_count") or 0),
                            reverse=True,
                        )[0]
                        fallback_type = "fallback"
                        break

            if not resolved_variant:
                continue

            fallback_rows.append({
                "disease_class_id": group["disease_class_id"],
                "clinical_doc_group_id": group["clinical_doc_group_id"],
                "specialty": group["specialty"],
                "disease_path": group["disease_path"],
                "doc_type": group["doc_type"],
                "requested_language": requested_lang,
                "resolved_language": resolved_lang,
                "fallback_type": fallback_type,
                "resolved_variant_id": resolved_variant.get("variant_id"),
                "resolved_source_format": resolved_variant.get("source_format"),
                "resolved_extraction_source": resolved_variant.get("extraction_source"),
                "resolved_char_count": resolved_variant.get("char_count"),
                "resolved_source_path": resolved_variant.get("source_path"),
            })

    return fallback_rows


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
        raise FileNotFoundError(
            f"No extraction rows found. Expected {RAW_JSONL} or {CLEANED_JSONL}"
        )

    quality_rows = [build_quality_row(row, args.min_chars) for row in rows]
    needs_review_rows = [r for r in quality_rows if r["needs_review"]]
    fallback_rows = build_fallback_after_extraction(rows)
    duplicate_rows = find_duplicates(rows)

    quality_fields = [
        "variant_id",
        "disease_class_id",
        "clinical_doc_group_id",
        "specialty",
        "disease_path",
        "doc_type",
        "source_path",
        "source_format",
        "extraction_source",
        "extraction_method",
        "language_hint",
        "detected_language",
        "final_language",
        "validation_text_source",
        "char_count",
        "letter_ratio",
        "digit_ratio",
        "cyrillic_ratio",
        "latin_ratio",
        "icd_code_count",
        "icd_codes_detected",
        "bad_pattern_count",
        "review_flag_count",
        "review_flags",
        "needs_review",
    ]

    fallback_fields = [
        "disease_class_id",
        "clinical_doc_group_id",
        "specialty",
        "disease_path",
        "doc_type",
        "requested_language",
        "resolved_language",
        "fallback_type",
        "resolved_variant_id",
        "resolved_source_format",
        "resolved_extraction_source",
        "resolved_char_count",
        "resolved_source_path",
    ]

    duplicate_fields = [
        "text_hash",
        "duplicate_group_size",
        "variant_id",
        "source_path",
        "specialty",
        "disease_path",
        "doc_type",
        "final_language",
        "char_count",
    ]

    write_csv(OUT_REPORT_CSV, quality_rows, quality_fields)
    write_csv(OUT_NEEDS_REVIEW_CSV, needs_review_rows, quality_fields)
    write_csv(OUT_LANGUAGE_FALLBACK_CSV, fallback_rows, fallback_fields)
    write_csv(OUT_DUPLICATES_CSV, duplicate_rows, duplicate_fields)

    summary = {
        "source_file": CLEANED_JSONL.as_posix() if args.prefer_cleaned and CLEANED_JSONL.exists() else RAW_JSONL.as_posix(),
        "records_total": len(rows),
        "needs_review_total": len(needs_review_rows),
        "needs_review_rate": round(len(needs_review_rows) / max(len(rows), 1), 4),
        "fallback_rows_total": len(fallback_rows),
        "possible_duplicate_rows": len(duplicate_rows),
        "by_source_format": dict(Counter(r["source_format"] for r in quality_rows)),
        "by_extraction_source": dict(Counter(r["extraction_source"] for r in quality_rows)),
        "by_doc_type": dict(Counter(r["doc_type"] for r in quality_rows)),
        "by_final_language": dict(Counter(r["final_language"] for r in quality_rows)),
        "review_flag_counts": dict(
            Counter(
                flag
                for r in quality_rows
                for flag in str(r["review_flags"]).split("; ")
                if flag
            )
        ),
        "icd_code_stats": {
            "records_with_icd": sum(1 for r in quality_rows if int(r["icd_code_count"]) > 0),
            "records_without_icd": sum(1 for r in quality_rows if int(r["icd_code_count"]) == 0),
        },
        "char_count_stats": {
            "min": min((int(r["char_count"]) for r in quality_rows), default=0),
            "max": max((int(r["char_count"]) for r in quality_rows), default=0),
            "avg": round(
                sum(int(r["char_count"]) for r in quality_rows) / max(len(quality_rows), 1),
                2,
            ),
        },
        "outputs": {
            "quality_report_json": OUT_REPORT_JSON.as_posix(),
            "quality_report_csv": OUT_REPORT_CSV.as_posix(),
            "needs_review_csv": OUT_NEEDS_REVIEW_CSV.as_posix(),
            "language_fallback_after_extraction_csv": OUT_LANGUAGE_FALLBACK_CSV.as_posix(),
            "possible_duplicates_csv": OUT_DUPLICATES_CSV.as_posix(),
        },
    }

    write_json(OUT_REPORT_JSON, summary)

    print("\nDONE")
    for k, v in summary.items():
        if k != "outputs":
            print(f"{k}: {v}")

    print("\nSaved:")
    for _, v in summary["outputs"].items():
        print(v)


if __name__ == "__main__":
    main()
