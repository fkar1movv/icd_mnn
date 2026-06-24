#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# 80_run_all.sh
# Full extraction pipeline runner
# Target: Ubuntu/Linux + RTX 5090 + prepared .venv_core/.venv_ocr
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="$LOG_DIR/pipeline_${TS}.log"

exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "============================================================"
echo "ICD/INN Extraction Pipeline"
echo "Started: $(date)"
echo "Project: $PROJECT_ROOT"
echo "Log: $PIPELINE_LOG"
echo "============================================================"

if [ ! -d "data/raw_guidelines" ]; then
  echo "ERROR: data/raw_guidelines not found"
  exit 1
fi

if [ -z "$(find data/raw_guidelines -type f | head -n 1)" ]; then
  echo "ERROR: data/raw_guidelines is empty"
  exit 1
fi

if [ ! -f "activate_core.sh" ]; then
  echo "ERROR: activate_core.sh not found. Run scripts/00_cloud_setup.sh first."
  exit 1
fi

if [ ! -f "activate_ocr.sh" ]; then
  echo "ERROR: activate_ocr.sh not found. Run scripts/00_cloud_setup.sh first."
  exit 1
fi

run_core() {
  echo ""
  echo "============================================================"
  echo "CORE STEP: $*"
  echo "============================================================"
  source activate_core.sh
  "$@"
}

run_ocr() {
  echo ""
  echo "============================================================"
  echo "OCR/GPU STEP: $*"
  echo "============================================================"
  source activate_ocr.sh
  "$@"
}

echo ""
echo "GPU check:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found"
fi

# ------------------------------------------------------------
# Step 1: Registry
# ------------------------------------------------------------

run_core python scripts/10_build_registry.py

# ------------------------------------------------------------
# Step 2: Word extraction
# ------------------------------------------------------------

WORD_WORKERS="${WORD_WORKERS:-8}"

run_core python scripts/20_extract_word.py --workers "$WORD_WORKERS"

# ------------------------------------------------------------
# Step 3: OCR queue
# ------------------------------------------------------------

QUEUE_WORKERS="${QUEUE_WORKERS:-4}"

run_core python scripts/30_build_ocr_queue.py --workers "$QUEUE_WORKERS"

# ------------------------------------------------------------
# Step 4: PaddleOCR-VL GPU OCR
# ------------------------------------------------------------

# RTX 5090 defaults. Override before running if needed:
# export GPU_MEMORY_UTILIZATION=0.85
# export MAX_NUM_BATCHED_TOKENS=16384

OCR_LIMIT="${OCR_LIMIT:-0}"

if [ "$OCR_LIMIT" = "0" ]; then
  run_ocr python scripts/40_run_paddleocr_vl.py
else
  run_ocr python scripts/40_run_paddleocr_vl.py --limit "$OCR_LIMIT"
fi

# ------------------------------------------------------------
# Step 5: Merge outputs
# ------------------------------------------------------------

run_core python scripts/50_merge_outputs.py

# ------------------------------------------------------------
# Step 6: LLM cleanup
# ------------------------------------------------------------

# Default: deterministic cleanup only.
# To use Gemma:
# export USE_LLM_CLEANUP=true in .env
# fill GEMMA_API_URL / GEMMA_API_KEY / GEMMA_MODEL
# or run 60_llm_cleanup.py manually with --use-llm

LLM_WORKERS="${LLM_WORKERS:-4}"

run_core python scripts/60_llm_cleanup.py --workers "$LLM_WORKERS"

# ------------------------------------------------------------
# Step 7: Validate
# ------------------------------------------------------------

run_core python scripts/70_validate_extraction.py
run_core python scripts/70_validate_extraction.py --prefer-cleaned

echo ""
echo "============================================================"
echo "PIPELINE FINISHED"
echo "Finished: $(date)"
echo "Log: $PIPELINE_LOG"
echo "============================================================"

echo ""
echo "Main outputs:"
echo "  data/outputs/document_registry_summary.json"
echo "  data/outputs/parsed_word_summary.json"
echo "  data/ocr_queue/ocr_queue_summary.json"
echo "  data/outputs/parsed_ocr_summary.json"
echo "  data/outputs/merge_outputs_summary.json"
echo "  data/outputs/llm_cleanup_summary.json"
echo "  data/outputs/extraction_quality_report.json"
echo ""
echo "Final corpus:"
echo "  data/outputs/parsed_guidelines_raw.jsonl"
echo "  data/outputs/parsed_guidelines_cleaned.jsonl"
echo ""
echo "Review files:"
echo "  data/outputs/needs_review.csv"
echo "  data/outputs/language_fallback_after_extraction.csv"
