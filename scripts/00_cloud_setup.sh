#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# 00_cloud_setup.sh
# Cloud setup for ICD/MKB → INN/MNN extraction pipeline
# Target: Ubuntu/Linux + RTX 5090 32GB + PyTorch CUDA container
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
# 3. .env
# ----------------------------

cat > .env <<'ENV_FILE'
RAW_GUIDELINES_DIR=data/raw_guidelines
OUTPUT_DIR=data/outputs
FAILED_DIR=data/failed
PARSED_WORD_DIR=data/parsed_word
PARSED_OCR_DIR=data/parsed_ocr
CONVERTED_DIR=data/converted
OCR_QUEUE_DIR=data/ocr_queue
LLM_CLEANED_DIR=data/llm_cleaned
LOG_DIR=data/logs

LIBREOFFICE_CMD=soffice

CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_BATCHED_TOKENS=32768
OCR_BATCH_SIZE=1
OCR_MAX_PAGES_PER_DOC=0
OCR_DPI=200

PADDLEOCR_VL_MODEL=PaddlePaddle/PaddleOCR-VL
PADDLEOCR_VL_SERVED_MODEL_NAME=PaddleOCR-VL-0.9B
PADDLEOCR_VL_PORT=8118

GEMMA_API_URL=
GEMMA_API_KEY=
USE_LLM_CLEANUP=false
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

python - <<'PY_CHECK_CORE'
import sys
import mammoth
import docx
import fitz
import pandas
import dotenv
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
python3 -m venv --system-site-packages .venv_ocr
source .venv_ocr/bin/activate

# DO NOT upgrade pip/setuptools/wheel here.
# In many CUDA containers, system wheel is Debian-managed and pip upgrade causes:
# "Cannot uninstall wheel ... RECORD file not found."

python -m pip install --ignore-installed --no-cache-dir \
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
  ftfy

echo ""
echo "Installing PaddleOCR package..."
python -m pip install --ignore-installed --no-cache-dir "paddleocr[doc-parser]" || {
  echo "WARNING: paddleocr[doc-parser] install failed."
  echo "Continuing setup; OCR install can be patched separately."
}

echo ""
echo "Installing PaddlePaddle GPU runtime..."
python -m pip install --ignore-installed --no-cache-dir paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ || {
  echo "WARNING: paddlepaddle-gpu cu126 install failed."
  echo "Trying default PyPI..."
  python -m pip install --ignore-installed --no-cache-dir paddlepaddle-gpu || {
    echo "WARNING: paddlepaddle-gpu install failed."
    echo "Continuing setup; OCR runtime can be patched separately."
  }
}

# vLLM is optional. Default OFF to avoid breaking setup.
# Enable later with: INSTALL_VLLM=1 bash scripts/00_cloud_setup.sh
if [ "${INSTALL_VLLM:-0}" = "1" ]; then
  echo ""
  echo "Installing vLLM nightly..."
  python -m pip install --no-cache-dir --pre vllm \
    --extra-index-url https://wheels.vllm.ai/nightly \
    --extra-index-url https://download.pytorch.org/whl/cu129 || {
      echo "WARNING: vLLM install failed. Continuing without vLLM."
    }
else
  echo "Skipping vLLM install. Set INSTALL_VLLM=1 to enable."
fi

python - <<'PY_CHECK_OCR'
import sys
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
    import paddle
    print("Paddle:", paddle.__version__)
    print("Paddle compiled with CUDA:", paddle.device.is_compiled_with_cuda())
    try:
        paddle.device.set_device("gpu:0")
        print("Paddle device:", paddle.device.get_device())
    except Exception as e:
        print("Paddle GPU device set failed:", repr(e))
except Exception as e:
    print("Paddle import failed:", repr(e))

try:
    import paddleocr
    print("PaddleOCR import OK")
except Exception as e:
    print("PaddleOCR import failed:", repr(e))

try:
    import vllm
    print("vLLM:", vllm.__version__)
except Exception as e:
    print("vLLM import skipped/failed:", repr(e))
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Paddle memory/performance flags
export FLAGS_fraction_of_gpu_memory_to_use=${GPU_MEMORY_UTILIZATION:-0.92}
export FLAGS_allocator_strategy=auto_growth
export FLAGS_cudnn_exhaustive_search=1

# Torch TF32 speed flags
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
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
echo "Next:"
echo "  source activate_core.sh"
echo "  python scripts/10_build_registry.py"
echo ""
echo "OCR:"
echo "  source activate_ocr.sh"
echo "  python scripts/40_run_paddleocr_vl.py"
