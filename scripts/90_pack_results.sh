#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
PACK_DIR="data/export/icd_inn_extraction_${TS}"
ZIP_PATH="data/export/icd_inn_extraction_${TS}.zip"

mkdir -p "$PACK_DIR"

echo "Packing extraction results..."
echo "Export dir: $PACK_DIR"

mkdir -p "$PACK_DIR/outputs"
mkdir -p "$PACK_DIR/failed"
mkdir -p "$PACK_DIR/logs"
mkdir -p "$PACK_DIR/ocr_queue"

copy_if_exists() {
  local src="$1"
  local dst="$2"

  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -r "$src" "$dst"
    echo "copied: $src"
  else
    echo "missing: $src"
  fi
}

# Main final corpus
copy_if_exists "data/outputs/parsed_guidelines_raw.jsonl" "$PACK_DIR/outputs/parsed_guidelines_raw.jsonl"
copy_if_exists "data/outputs/parsed_guidelines_cleaned.jsonl" "$PACK_DIR/outputs/parsed_guidelines_cleaned.jsonl"
copy_if_exists "data/outputs/parsed_guidelines_grouped_raw.json" "$PACK_DIR/outputs/parsed_guidelines_grouped_raw.json"

# Registry
copy_if_exists "data/outputs/document_registry.jsonl" "$PACK_DIR/outputs/document_registry.jsonl"
copy_if_exists "data/outputs/document_registry_grouped.json" "$PACK_DIR/outputs/document_registry_grouped.json"
copy_if_exists "data/outputs/document_registry_selected.csv" "$PACK_DIR/outputs/document_registry_selected.csv"
copy_if_exists "data/outputs/language_fallback_plan.csv" "$PACK_DIR/outputs/language_fallback_plan.csv"
copy_if_exists "data/outputs/language_fallback_after_extraction.csv" "$PACK_DIR/outputs/language_fallback_after_extraction.csv"

# Summaries
copy_if_exists "data/outputs/document_registry_summary.json" "$PACK_DIR/outputs/document_registry_summary.json"
copy_if_exists "data/outputs/parsed_word_summary.json" "$PACK_DIR/outputs/parsed_word_summary.json"
copy_if_exists "data/ocr_queue/ocr_queue_summary.json" "$PACK_DIR/ocr_queue/ocr_queue_summary.json"
copy_if_exists "data/outputs/parsed_ocr_summary.json" "$PACK_DIR/outputs/parsed_ocr_summary.json"
copy_if_exists "data/outputs/merge_outputs_summary.json" "$PACK_DIR/outputs/merge_outputs_summary.json"
copy_if_exists "data/outputs/llm_cleanup_summary.json" "$PACK_DIR/outputs/llm_cleanup_summary.json"
copy_if_exists "data/outputs/extraction_quality_report.json" "$PACK_DIR/outputs/extraction_quality_report.json"

# Review CSVs
copy_if_exists "data/outputs/extraction_quality_report.csv" "$PACK_DIR/outputs/extraction_quality_report.csv"
copy_if_exists "data/outputs/needs_review.csv" "$PACK_DIR/outputs/needs_review.csv"
copy_if_exists "data/outputs/missing_extracted_variants.csv" "$PACK_DIR/outputs/missing_extracted_variants.csv"
copy_if_exists "data/outputs/possible_duplicate_texts.csv" "$PACK_DIR/outputs/possible_duplicate_texts.csv"

# Queue and failure logs
copy_if_exists "data/ocr_queue/ocr_queue.csv" "$PACK_DIR/ocr_queue/ocr_queue.csv"
copy_if_exists "data/ocr_queue/ocr_queue.jsonl" "$PACK_DIR/ocr_queue/ocr_queue.jsonl"

copy_if_exists "data/failed/word_failures.jsonl" "$PACK_DIR/failed/word_failures.jsonl"
copy_if_exists "data/failed/ocr_failures.jsonl" "$PACK_DIR/failed/ocr_failures.jsonl"
copy_if_exists "data/failed/ocr_queue_conversion_failures.jsonl" "$PACK_DIR/failed/ocr_queue_conversion_failures.jsonl"
copy_if_exists "data/failed/llm_cleanup_failures.jsonl" "$PACK_DIR/failed/llm_cleanup_failures.jsonl"
copy_if_exists "data/failed/registry_skipped_files.csv" "$PACK_DIR/failed/registry_skipped_files.csv"

# Logs
if [ -d "data/logs" ]; then
  cp -r data/logs/* "$PACK_DIR/logs/" 2>/dev/null || true
fi

# Pack metadata
cat > "$PACK_DIR/export_manifest.json" <<EOF
{
  "created_at": "$(date -Iseconds)",
  "project_root": "$PROJECT_ROOT",
  "contains_raw_guidelines": false,
  "contains_parsed_text": true,
  "contains_registry": true,
  "contains_quality_reports": true,
  "contains_failure_logs": true
}
EOF

mkdir -p data/export

if command -v zip >/dev/null 2>&1; then
  cd data/export
  zip -r "$(basename "$ZIP_PATH")" "$(basename "$PACK_DIR")"
  cd "$PROJECT_ROOT"
else
  tar -czf "${ZIP_PATH%.zip}.tar.gz" -C data/export "$(basename "$PACK_DIR")"
  ZIP_PATH="${ZIP_PATH%.zip}.tar.gz"
fi

echo ""
echo "DONE"
echo "Export folder: $PACK_DIR"
echo "Archive: $ZIP_PATH"
