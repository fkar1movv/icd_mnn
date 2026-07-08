#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
PACK_DIR="data/export/pdf_agentic_extraction_${TS}"
ZIP_PATH="data/export/pdf_agentic_extraction_${TS}.zip"

mkdir -p "$PACK_DIR"
mkdir -p "$PACK_DIR/outputs"
mkdir -p "$PACK_DIR/parsed_docling"
mkdir -p "$PACK_DIR/parsed_pdf_agentic"
mkdir -p "$PACK_DIR/parsed_pdf_docling_raw"
mkdir -p "$PACK_DIR/pdf_visual_assets"

echo "Packing PDF-only extraction outputs..."
echo "Export dir: $PACK_DIR"

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

# ============================================================
# Final extraction JSONL / grouped JSON / validation reports
# ============================================================

copy_if_exists "data/outputs/parsed_guidelines_raw.jsonl" "$PACK_DIR/outputs/parsed_guidelines_raw.jsonl"
copy_if_exists "data/outputs/parsed_guidelines_grouped_raw.json" "$PACK_DIR/outputs/parsed_guidelines_grouped_raw.json"
copy_if_exists "data/outputs/merge_outputs_summary.json" "$PACK_DIR/outputs/merge_outputs_summary.json"

copy_if_exists "data/outputs/parsed_docling.jsonl" "$PACK_DIR/outputs/parsed_docling.jsonl"
copy_if_exists "data/outputs/parsed_docling_summary.json" "$PACK_DIR/outputs/parsed_docling_summary.json"

copy_if_exists "data/outputs/pdf_visual_manifest.jsonl" "$PACK_DIR/outputs/pdf_visual_manifest.jsonl"
copy_if_exists "data/outputs/pdf_visual_manifest.csv" "$PACK_DIR/outputs/pdf_visual_manifest.csv"

copy_if_exists "data/outputs/extraction_quality_report.json" "$PACK_DIR/outputs/extraction_quality_report.json"
copy_if_exists "data/outputs/extraction_quality_report.csv" "$PACK_DIR/outputs/extraction_quality_report.csv"
copy_if_exists "data/outputs/needs_review.csv" "$PACK_DIR/outputs/needs_review.csv"
copy_if_exists "data/outputs/pending_visual_assets_review.csv" "$PACK_DIR/outputs/pending_visual_assets_review.csv"
copy_if_exists "data/outputs/possible_duplicate_texts.csv" "$PACK_DIR/outputs/possible_duplicate_texts.csv"

# ============================================================
# Per-PDF extraction artifacts
# JSON + Markdown only.
# ============================================================

copy_if_exists "data/parsed_docling" "$PACK_DIR/parsed_docling"
copy_if_exists "data/parsed_pdf_agentic" "$PACK_DIR/parsed_pdf_agentic"
copy_if_exists "data/parsed_pdf_docling_raw" "$PACK_DIR/parsed_pdf_docling_raw"

# ============================================================
# Visual assets: only manifests + complex crops.
# Do NOT pack rendered full page images.
# ============================================================

if [ -d "data/pdf_visual_assets" ]; then
  while IFS= read -r -d '' manifest; do
    rel="${manifest#data/pdf_visual_assets/}"
    mkdir -p "$(dirname "$PACK_DIR/pdf_visual_assets/$rel")"
    cp "$manifest" "$PACK_DIR/pdf_visual_assets/$rel"
  done < <(find data/pdf_visual_assets -name "visual_manifest.jsonl" -print0)

  while IFS= read -r -d '' cropdir; do
    rel="${cropdir#data/pdf_visual_assets/}"
    mkdir -p "$PACK_DIR/pdf_visual_assets/$rel"
    cp -r "$cropdir"/* "$PACK_DIR/pdf_visual_assets/$rel/" 2>/dev/null || true
  done < <(find data/pdf_visual_assets -type d -name "crops" -print0)
fi

# ============================================================
# Export manifest
# ============================================================

cat > "$PACK_DIR/export_manifest.json" <<EOF
{
  "created_at": "$(date -Iseconds)",
  "project_root": "$PROJECT_ROOT",
  "pipeline": "pdf_only_agentic_docling_surya_deferred_vlm",
  "contains_raw_guidelines": false,
  "contains_word_outputs": false,
  "contains_pdf_extraction_json": true,
  "contains_markdown": true,
  "contains_docling_json": true,
  "contains_rendered_page_images": false,
  "contains_complex_visual_crops": true,
  "contains_validation_reports": true,
  "vlm_status": "deferred"
}
EOF

# ============================================================
# Zip
# ============================================================

mkdir -p data/export

if command -v zip >/dev/null 2>&1; then
  (cd "$(dirname "$PACK_DIR")" && zip -qr "$(basename "$ZIP_PATH")" "$(basename "$PACK_DIR")")
  echo "ZIP created: $ZIP_PATH"
else
  echo "zip command not found; folder export is ready: $PACK_DIR"
fi

echo "PACK DONE"
