#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# 00_cloud_setup.sh
# Cloud setup for ICD/MKB → INN/MNN extraction pipeline
#
# New extraction stack:
#   PDF      → Docling + SuryaOCR
#   DOC/DOCX → native Python extractors + Docling fallback later
#
# Target:
#   Vast.ai Ubuntu/Linux + NVIDIA GPU + PyTorch CUDA container
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

# ----------------------------
# 1. System packages
# ----------------------------

if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then
    APT="apt-get"
  elif command -v sudo >/dev/null 2>&1; then
    APT="sudo apt-get"
  else
    APT=""
  fi

  if [ -n "$APT" ]; then
    echo "Installing system packages..."
    $APT update

    $APT install -y \
      libreoffice \
      libreoffice-writer \
      poppler-utils \
      git \
      curl \
      wget \
      unzip \
      zip \
      rsync \
      jq \
      dos2unix \
      build-essential \
      python3 \
      python3-pip \
      python3-venv \
      python3-dev \
      python-is-python3 \
      fonts-dejavu \
      fonts-liberation \
      fonts-noto-core \
      fonts-noto-cjk \
      fontconfig \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1
  else
    echo "No root/sudo access. Skipping apt install."
  fi
else
  echo "apt-get not found. Skipping apt install."
fi

# ----------------------------
# 2. Folder structure
# ----------------------------

mkdir -p \
  data/raw_guidelines \
  data/outputs \
  data/failed \
  data/parsed_word \
  data/parsed_docling \
  data/pdf_work \
  data/pdf_visual_assets \
  data/parsed_pdf_agentic \
  data/parsed_pdf_docling_raw \
  data/pdf_visual_results \
  data/converted \
  data/ocr_queue \
  data/logs \
  data/export \
  models \
  tmp \
  src \
  scripts

touch src/__init__.py

# ----------------------------
# 3. .env
# ----------------------------

cat > .env <<'ENV_FILE'
RAW_GUIDELINES_DIR=data/raw_guidelines
OUTPUT_DIR=data/outputs
FAILED_DIR=data/failed
PARSED_WORD_DIR=data/parsed_word
PARSED_DOCLING_DIR=data/parsed_docling
PARSED_OCR_DIR=data/parsed_docling

PDF_WORK_DIR=data/pdf_work
PDF_VISUAL_ASSETS_DIR=data/pdf_visual_assets
PARSED_PDF_AGENTIC_DIR=data/parsed_pdf_agentic
PARSED_PDF_DOCLING_RAW_DIR=data/parsed_pdf_docling_raw
PDF_VISUAL_RESULTS_DIR=data/pdf_visual_results

CONVERTED_DIR=data/converted
OCR_QUEUE_DIR=data/ocr_queue
LOG_DIR=data/logs

LIBREOFFICE_CMD=soffice

CUDA_VISIBLE_DEVICES=0

# Docling + SuryaOCR
OCR_ENGINE=docling_surya
DOCLING_OCR_LANGS=uz,ru,en
DOCLING_FORCE_FULL_PAGE_OCR=true
DOCLING_DO_TABLE_STRUCTURE=true
DOCLING_GENERATE_PAGE_IMAGES=false
DOCLING_GENERATE_PICTURE_IMAGES=false
DOCLING_IMAGES_SCALE=1.0
DOCLING_NUM_THREADS=1
OMP_NUM_THREADS=1

# Agentic PDF extraction
PDF_BATCH_SIZE=8
PDF_RENDER_DPI=220
PDF_CROP_PADDING_PX=12
PDF_KEEP_PAGE_IMAGES=false
PDF_KEEP_WORK_DIR=false
PDF_MIN_CHARS=500
PDF_DEFER_VLM=true

# Surya runtime
# On NVIDIA GPU, Surya 2 can use vLLM.
# If vLLM is unavailable, extraction script should fail gracefully and log the error.
SURYA_INFERENCE_BACKEND=vllm
SURYA_INFERENCE_KEEP_ALIVE=1
SURYA_INFERENCE_PARALLEL=1

# Memory-safety defaults
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TOKENIZERS_PARALLELISM=false
NVIDIA_TF32_OVERRIDE=1
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# vLLM install is attempted by default.
# Set INSTALL_VLLM=0 before running setup to skip it.
INSTALL_VLLM=1
ENV_FILE

echo "Created .env"

# ----------------------------
# 4. Core Python env
# ----------------------------

echo ""
echo "========== CORE ENV =========="

rm -rf .venv_core
python3 -m venv .venv_core
source .venv_core/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install --no-cache-dir \
  mammoth \
  python-docx \
  docx2python \
  beautifulsoup4 \
  lxml \
  ftfy \
  pymupdf \
  pandas \
  tqdm \
  jsonschema \
  python-dotenv \
  httpx \
  tenacity \
  orjson \
  regex \
  docling

python - <<'PY_CHECK_CORE'
import sys
import mammoth
import docx
import fitz
import pandas
import dotenv
import docling
from docx2python import docx2python

print("CORE ENV OK")
print("Python:", sys.version)
PY_CHECK_CORE

deactivate

# ----------------------------
# 5. OCR/GPU Python env
# ----------------------------

echo ""
echo "========== OCR ENV =========="

rm -rf .venv_ocr

# Keep system CUDA/Torch visible from Vast.ai PyTorch container.
# Important: do NOT use --ignore-installed here, otherwise pip may replace
# the container's working CUDA torch stack.
python3 -m venv --system-site-packages .venv_ocr
source .venv_ocr/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install --no-cache-dir \
  pillow \
  pymupdf \
  pandas \
  tqdm \
  python-dotenv \
  httpx \
  tenacity \
  orjson \
  regex \
  beautifulsoup4 \
  lxml \
  ftfy \
  docling \
  docling-surya \
  surya-ocr

# vLLM is useful for Surya 2 on NVIDIA GPU, but it is the most fragile dependency.
# Keep setup non-blocking: if vLLM fails, the extraction script will still log clear failures.
INSTALL_VLLM="${INSTALL_VLLM:-1}"

if [ "$INSTALL_VLLM" = "1" ]; then
  echo ""
  echo "Installing vLLM..."

  python -m pip install --no-cache-dir vllm || {
    echo "WARNING: vLLM install failed with default PyPI."
    echo "Trying vLLM nightly wheels..."

    python -m pip install --no-cache-dir --pre vllm \
      --extra-index-url https://wheels.vllm.ai/nightly \
      --extra-index-url https://download.pytorch.org/whl/cu129 || {
        echo "WARNING: vLLM install failed. Continuing without vLLM."
      }
  }
else
  echo "Skipping vLLM install because INSTALL_VLLM=0."
fi

python - <<'PY_CHECK_OCR'
import sys
import importlib.util

print("OCR ENV CHECK")
print("Python:", sys.version)

try:
    import torch
    print("Torch:", torch.__version__)
    print("Torch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
except Exception as e:
    print("Torch check failed:", repr(e))

try:
    import docling
    print("Docling import OK")
except Exception as e:
    print("Docling import failed:", repr(e))

try:
    from docling_surya import SuryaOcrOptions
    print("docling-surya import OK")
except Exception as e:
    print("docling-surya import failed:", repr(e))

try:
    import surya
    print("surya import OK")
except Exception as e:
    print("surya import failed:", repr(e))

try:
    import vllm
    print("vLLM:", vllm.__version__)
except Exception as e:
    print("vLLM import skipped/failed:", repr(e))

print("vLLM available:", importlib.util.find_spec("vllm") is not None)
PY_CHECK_OCR

deactivate

# ----------------------------
# 6. GPU check
# ----------------------------

echo ""
echo "========== GPU CHECK =========="

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found."
fi

# ----------------------------
# 7. Activation helpers
# ----------------------------

cat > activate_core.sh <<'ACT_CORE'
#!/usr/bin/env bash
source .venv_core/bin/activate
set -a
source .env
set +a
ACT_CORE

cat > activate_ocr.sh <<'ACT_OCR'
#!/usr/bin/env bash
source .venv_ocr/bin/activate
set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Memory safety
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# Docling / native preprocessing stability
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export DOCLING_NUM_THREADS=${DOCLING_NUM_THREADS:-1}

# Surya runtime
export SURYA_INFERENCE_BACKEND=${SURYA_INFERENCE_BACKEND:-vllm}
export SURYA_INFERENCE_KEEP_ALIVE=${SURYA_INFERENCE_KEEP_ALIVE:-1}
export SURYA_INFERENCE_PARALLEL=${SURYA_INFERENCE_PARALLEL:-1}

# Torch speed flags
export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-1}
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}
ACT_OCR

chmod +x activate_core.sh activate_ocr.sh

# ----------------------------
# 8. Final command checks
# ----------------------------

echo ""
echo "========== COMMAND CHECKS =========="

echo "LibreOffice:"
if command -v soffice >/dev/null 2>&1; then
  soffice --version || true
else
  echo "soffice not found"
fi

echo ""
echo "Core Python:"
source .venv_core/bin/activate
python --version
deactivate

echo ""
echo "OCR Python:"
source .venv_ocr/bin/activate
python --version
deactivate

echo ""
echo "SETUP DONE"
echo ""
echo "Next commands:"
echo "  source activate_core.sh"
echo "  python scripts/10_build_registry.py"
echo ""
echo "Word extraction:"
echo "  source activate_core.sh"
echo "  python scripts/20_extract_word.py"
echo ""
echo "PDF extraction with Docling + SuryaOCR:"
echo "  source activate_ocr.sh"
echo "  python scripts/40_run_docling_surya.py"


echo ""
echo "PDF-only agentic pipeline commands:"
echo "  source activate_core.sh"
echo "  python scripts/10_build_registry.py"
echo "  python scripts/30_build_ocr_queue.py --include-backfill-pdfs"
echo "  source activate_ocr.sh"
echo "  python scripts/40_run_doclingocr.py"
echo "  source activate_core.sh"
echo "  python scripts/50_merge_outputs.py"
echo "  python scripts/60_validate_extraction.py"
echo "  bash scripts/pack_results.sh"
