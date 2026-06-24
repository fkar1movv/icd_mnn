#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# 00_cloud_setup.sh
# Cloud setup for ICD/MKB → INN/MNN extraction pipeline
# Target: Ubuntu/Linux + NVIDIA RTX 5090 32GB + PyTorch CUDA container
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

# ----------------------------
# 1. System packages
# ----------------------------

install_apt_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get not found. Skipping system package install."
    return 0
  fi

  APT_CMD="apt-get"
  if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
      APT_CMD="sudo apt-get"
    else
      echo "Not root and sudo not found. Skipping apt install."
      return 0
    fi
  fi

  echo "Installing system packages..."

  $APT_CMD update

  DEBIAN_FRONTEND=noninteractive $APT_CMD install -y \
    libreoffice \
    libreoffice-writer \
    poppler-utils \
    git \
    curl \
    wget \
    unzip \
    rsync \
    jq \
    build-essential \
    python3 \
    python3-venv \
    python3-dev \
    fonts-dejavu \
    fonts-liberation \
    fontconfig \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

  echo "System packages installed."
}

install_apt_packages

# ----------------------------
# 2. Folder structure
# ----------------------------

mkdir -p \
  data/raw_guidelines \
  data/outputs \
  data/failed \
  data/parsed_word \
  data/parsed_ocr \
  data/converted \
  data/ocr_queue \
  data/llm_cleaned \
  data/logs \
  models \
  tmp \
  src \
  scripts

touch src/__init__.py

# ----------------------------
# 3. Environment config
# ----------------------------

if [ ! -f ".env" ]; then

# System commands
LIBREOFFICE_CMD=soffice

# GPU / OCR settings
OCR_BATCH_SIZE=1
OCR_MAX_PAGES_PER_DOC=0
OCR_DPI=200

# PaddleOCR-VL / vLLM

# LLM cleanup API; fill later
EOF
  echo "Created .env"
else
  echo ".env already exists; not overwriting."
fi

# ----------------------------
# 4. Python env: core pipeline
# ----------------------------

echo "Creating .venv_core..."

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
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
  regex

python - <<'PY'
import sys
import mammoth
import docx
import fitz
import pandas
import dotenv
print("CORE ENV OK")
print("Python:", sys.version)
PY

deactivate

# ----------------------------
# 5. Python env: OCR / GPU pipeline
# ----------------------------

echo "Creating .venv_ocr..."

python3 -m venv .venv_ocr
source .venv_ocr/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Basic OCR utilities
python -m pip install \
  pillow \
  pymupdf \
  pandas \
  tqdm \
  python-dotenv \
  httpx \
  tenacity \
  orjson \
  regex

# PaddleOCR document parser package.
# PaddleOCR docs use optional dependency groups for document parsing.
python -m pip install -U "paddleocr[doc-parser]"

# vLLM setup for PaddleOCR-VL.
# For RTX 50-series / Blackwell, CUDA 12.8/12.9-compatible wheels are usually needed.
# vLLM's PaddleOCR-VL recipe currently uses nightly vLLM wheels with CUDA 12.9 index.
python -m pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match || {
    echo "vLLM install failed."
    echo "This may happen if the container Python/CUDA version is incompatible."
    echo "We can patch this after seeing the exact error."
    exit 1
  }

python - <<'PY'
import sys
print("OCR ENV BASIC OK")
print("Python:", sys.version)

try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
except Exception as e:
    print("Torch check failed:", repr(e))

try:
    import vllm
    print("vLLM:", vllm.__version__)
except Exception as e:
    print("vLLM import failed:", repr(e))

try:
    import paddleocr
    print("PaddleOCR import OK")
except Exception as e:
    print("PaddleOCR import failed:", repr(e))
PY

deactivate

# ----------------------------
# 6. GPU sanity check from container
# ----------------------------

echo ""
echo "========== GPU CHECK =========="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found. Check Vast.ai container GPU passthrough."
fi

# ----------------------------
# 7. Create helper runner files
# ----------------------------

cat > activate_core.sh <<'EOF'
#!/usr/bin/env bash
source .venv_core/bin/activate
export $(grep -v '^#' .env | xargs -d '\n' 2>/dev/null || true)
EOF

cat > activate_ocr.sh <<'EOF'
#!/usr/bin/env bash
source .venv_ocr/bin/activate
export $(grep -v '^#' .env | xargs -d '\n' 2>/dev/null || true)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
EOF

chmod +x activate_core.sh activate_ocr.sh

# ----------------------------
# 8. Final checks
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
echo "Python core:"
source .venv_core/bin/activate
python --version
deactivate

echo ""
echo "Python OCR:"
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
echo "For OCR:"
echo "  source activate_ocr.sh"
echo "  python scripts/40_run_paddleocr_vl.py"
python3 -m venv .venv_core
source .venv_core/bin/activate
GEMMA_API_URL=
GEMMA_API_KEY=
USE_LLM_CLEANUP=false
PADDLEOCR_VL_MODEL=PaddlePaddle/PaddleOCR-VL
PADDLEOCR_VL_SERVED_MODEL_NAME=PaddleOCR-VL-0.9B
PADDLEOCR_VL_PORT=8118
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_BATCHED_TOKENS=32768
OCR_QUEUE_DIR=data/ocr_queue
LLM_CLEANED_DIR=data/llm_cleaned
LOG_DIR=data/logs
PARSED_WORD_DIR=data/parsed_word
PARSED_OCR_DIR=data/parsed_ocr
CONVERTED_DIR=data/converted
  cat > .env <<'EOF'
# Paths
FAILED_DIR=data/failed

