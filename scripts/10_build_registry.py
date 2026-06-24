#!/usr/bin/env python3
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    RAW_GUIDELINES_DIR,
    OUTPUT_DIR,
    FAILED_DIR,
    WORD_EXTENSIONS,
    PDF_EXTENSIONS,
    REQUEST_LANGUAGES,
    FALLBACK_LANGUAGE_PRIORITY,
    build_base_variant_record,
    ensure_base_dirs,
    is_supported_file,
    is_temp_word_file,
    is_review_path,
    write_json,
    write_jsonl,
    write_csv,
)


EXT_PRIORITY = {
    ".docx": 1,
    ".doc": 2,
    ".pdf": 3,
}


OUT_REGISTRY_JSONL = OUTPUT_DIR / "document_registry.jsonl"
OUT_GROUPED_JSON = OUTPUT_DIR / "document_registry_grouped.json"
OUT_SELECTED_CSV = OUTPUT_DIR / "document_registry_selected.csv"
OUT_FALLBACK_CSV = OUTPUT_DIR / "language_fallback_plan.csv"
OUT_PDF_BACKFILL_CSV = OUTPUT_DIR / "pdf_backfill_candidates.csv"
OUT_INVENTORY_JSON = OUTPUT_DIR / "document_inventory.json"
OUT_SUMMARY_JSON = OUTPUT_DIR / "document_registry_summary.json"
OUT_SKIPPED_CSV = FAILED_DIR / "registry_skipped_files.csv"


def scan_files(raw_dir: Path) -> Tuple[List[Path], List[Dict[str, Any]]]:
    files = []
    skipped = []

    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue

        reason = None

        if is_temp_word_file(path):
            reason = "temporary_word_lock_file"
        elif not is_supported_file(path):
            reason = "unsupported_extension"

        if reason:
            skipped.append({
                "path": path.as_posix(),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "skip_reason": reason,
            })
            continue

        files.append(path)

    return files, skipped


def choose_best(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(
        items,
        key=lambda x: (
            x["priority"],
            -float(x.get("size_kb") or 0),
            x["source_path"],
        )
    )[0]


def select_group_variants(items: List[Dict[str, Any]]):
    """
    Selection rule:
    - group = disease_class_id + doc_type
    - if group has any Word file, select Word only
    - if no Word exists, select PDFs
    - known languages: select one best file per ru/uz_cyr/uz_lat
    - unknown language: keep one best per normalized_file_key for later content detection
    """

    selected = []
    rejected = []
    pdf_backfill = []

    word_items = [x for x in items if x["source_extension"] in WORD_EXTENSIONS]
    pdf_items = [x for x in items if x["source_extension"] in PDF_EXTENSIONS]

    if word_items:
        active_pool = word_items
        pool_reason = "word_primary"
        for p in pdf_items:
            p2 = dict(p)
            p2["selected"] = False
            p2["selection_reason"] = "pdf_backfill_candidate_word_exists"
            pdf_backfill.append(p2)
    else:
        active_pool = pdf_items
        pool_reason = "pdf_primary_no_word_available"

    # Known languages: one best file per language.
    for lang in ["ru", "uz_cyr", "uz_lat"]:
        lang_items = [x for x in active_pool if x["language_hint"] == lang]
        if not lang_items:
            continue

        best = choose_best(lang_items)
        best["selected"] = True
        best["selection_reason"] = f"{pool_reason}_best_for_language_{lang}"
        selected.append(best)

        for item in lang_items:
            if item["variant_id"] != best["variant_id"]:
                dup = dict(item)
                dup["selected"] = False
                dup["selection_reason"] = f'duplicate_of_{best["variant_id"]}'
                rejected.append(dup)

    # Unknown-language files: preserve different files but deduplicate near-identical names.
    unknown_items = [x for x in active_pool if x["language_hint"] == "unknown"]

    unknown_groups = defaultdict(list)
    for item in unknown_items:
        unknown_groups[item["normalized_file_key"]].append(item)

    for _, subitems in unknown_groups.items():
        best = choose_best(subitems)
        best["selected"] = True
        best["selection_reason"] = f"{pool_reason}_unknown_language_keep_for_content_detection"
        selected.append(best)

        for item in subitems:
            if item["variant_id"] != best["variant_id"]:
                dup = dict(item)
                dup["selected"] = False
                dup["selection_reason"] = f'duplicate_of_{best["variant_id"]}'
                rejected.append(dup)

    return selected, rejected, pdf_backfill


def build_grouped_registry(selected_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(lambda: {
        "disease_class_id": None,
        "clinical_doc_group_id": None,
        "specialty": None,
        "disease_path": None,
        "doc_type": None,
        "available_languages": [],
        "variants": defaultdict(list),
    })

    for row in selected_rows:
        key = (row["disease_class_id"], row["doc_type"])
        g = grouped[key]

        g["disease_class_id"] = row["disease_class_id"]
        g["clinical_doc_group_id"] = row["clinical_doc_group_id"]
        g["specialty"] = row["specialty"]
        g["disease_path"] = row["disease_path"]
        g["doc_type"] = row["doc_type"]
        g["variants"][row["language_hint"]].append(row)

    result = []

    for _, g in grouped.items():
        variants = {lang: rows for lang, rows in g["variants"].items()}
        result.append({
            "disease_class_id": g["disease_class_id"],
            "clinical_doc_group_id": g["clinical_doc_group_id"],
            "specialty": g["specialty"],
            "disease_path": g["disease_path"],
            "doc_type": g["doc_type"],
            "available_languages": sorted(variants.keys()),
            "variants": variants,
        })

    return sorted(result, key=lambda x: (x["specialty"], x["disease_path"], x["doc_type"]))


def build_language_fallback_plan(grouped_registry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for group in grouped_registry:
        variants = group["variants"]

        for requested_lang in REQUEST_LANGUAGES:
            resolved_lang = None
            resolved_variant = None
            fallback_type = None

            if requested_lang in variants:
                resolved_lang = requested_lang
                resolved_variant = variants[requested_lang][0]
                fallback_type = "exact"
            else:
                for lang in FALLBACK_LANGUAGE_PRIORITY:
                    if lang in variants:
                        resolved_lang = lang
                        resolved_variant = variants[lang][0]
                        fallback_type = "fallback"
                        break

            if not resolved_variant:
                continue

            rows.append({
                "disease_class_id": group["disease_class_id"],
                "clinical_doc_group_id": group["clinical_doc_group_id"],
                "specialty": group["specialty"],
                "disease_path": group["disease_path"],
                "doc_type": group["doc_type"],
                "requested_language": requested_lang,
                "resolved_language": resolved_lang,
                "fallback_type": fallback_type,
                "resolved_variant_id": resolved_variant["variant_id"],
                "resolved_source_format": resolved_variant["source_format"],
                "resolved_source_path": resolved_variant["source_path"],
            })

    return rows


def make_inventory_record(path: Path, idx: int) -> Dict[str, Any]:
    row = build_base_variant_record(path)
    row["inventory_id"] = f"inv_{idx:06d}"
    row["priority"] = EXT_PRIORITY.get(row["source_extension"], 99)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default=str(RAW_GUIDELINES_DIR))
    parser.add_argument("--include-reviews", action="store_true")
    args = parser.parse_args()

    ensure_base_dirs()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw guidelines directory not found: {raw_dir}")

    print(f"Scanning: {raw_dir}")

    files, skipped = scan_files(raw_dir)

    inventory = []
    excluded_reviews = []

    for idx, path in enumerate(sorted(files), start=1):
        if not args.include_reviews and is_review_path(path):
            excluded_reviews.append({
                "path": path.as_posix(),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "skip_reason": "review_or_expert_document",
            })
            continue

        row = make_inventory_record(path, idx)
        inventory.append(row)

    # group by disease + doc_type
    groups = defaultdict(list)
    for row in inventory:
        key = (row["disease_class_id"], row["doc_type"])
        groups[key].append(row)

    selected = []
    rejected_duplicates = []
    pdf_backfill = []

    for _, items in groups.items():
        s, r, p = select_group_variants(items)
        selected.extend(s)
        rejected_duplicates.extend(r)
        pdf_backfill.extend(p)

    selected = sorted(
        selected,
        key=lambda x: (
            x["specialty"],
            x["disease_path"],
            x["doc_type"],
            x["language_hint"],
            x["priority"],
            x["source_path"],
        )
    )

    grouped_registry = build_grouped_registry(selected)
    fallback_plan = build_language_fallback_plan(grouped_registry)

    selected_fields = [
        "variant_id",
        "inventory_id",
        "disease_class_id",
        "clinical_doc_group_id",
        "specialty",
        "disease_path",
        "doc_type",
        "language_hint",
        "source_format",
        "source_extension",
        "priority",
        "size_kb",
        "normalized_file_key",
        "source_path",
        "selection_reason",
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
        "resolved_source_path",
    ]

    skipped_all = skipped + excluded_reviews

    write_json(OUT_INVENTORY_JSON, inventory)
    write_jsonl(OUT_REGISTRY_JSONL, selected)
    write_json(OUT_GROUPED_JSON, grouped_registry)
    write_csv(OUT_SELECTED_CSV, selected, selected_fields)
    write_csv(OUT_FALLBACK_CSV, fallback_plan, fallback_fields)
    write_csv(OUT_PDF_BACKFILL_CSV, pdf_backfill, selected_fields)
    write_csv(OUT_SKIPPED_CSV, skipped_all)

    summary = {
        "raw_dir": raw_dir.as_posix(),
        "files_scanned_supported": len(files),
        "inventory_total_after_review_filter": len(inventory),
        "selected_total": len(selected),
        "rejected_duplicates_total": len(rejected_duplicates),
        "pdf_backfill_candidates_total": len(pdf_backfill),
        "skipped_total": len(skipped),
        "excluded_reviews_total": len(excluded_reviews),
        "disease_classes_total": len(set(x["disease_class_id"] for x in selected)),
        "clinical_doc_groups_total": len(grouped_registry),
        "fallback_rows_total": len(fallback_plan),
        "selected_by_extension": dict(Counter(x["source_extension"] for x in selected)),
        "selected_by_source_format": dict(Counter(x["source_format"] for x in selected)),
        "selected_by_language_hint": dict(Counter(x["language_hint"] for x in selected)),
        "selected_by_doc_type": dict(Counter(x["doc_type"] for x in selected)),
        "fallback_type_counts": dict(Counter(x["fallback_type"] for x in fallback_plan)),
        "outputs": {
            "inventory_json": OUT_INVENTORY_JSON.as_posix(),
            "registry_jsonl": OUT_REGISTRY_JSONL.as_posix(),
            "grouped_registry_json": OUT_GROUPED_JSON.as_posix(),
            "selected_csv": OUT_SELECTED_CSV.as_posix(),
            "fallback_csv": OUT_FALLBACK_CSV.as_posix(),
            "pdf_backfill_csv": OUT_PDF_BACKFILL_CSV.as_posix(),
            "skipped_csv": OUT_SKIPPED_CSV.as_posix(),
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
