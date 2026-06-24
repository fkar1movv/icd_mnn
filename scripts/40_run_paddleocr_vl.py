#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from tqdm import tqdm

from src.common import (
    FAILED_DIR,
    OCR_QUEUE_DIR,
    OUTPUT_DIR,
    PARSED_OCR_DIR,
    append_jsonl,
    clean_text,
    detect_language_from_text,
    ensure_base_dirs,
    get_quality_flags,
    iter_jsonl,
    read_jsonl,
    stable_id,
    write_json,
    write_jsonl,
)


QUEUE_JSONL = OCR_QUEUE_DIR / "ocr_queue.jsonl"

OUT_PARSED_OCR_JSONL = OUTPUT_DIR / "parsed_ocr.jsonl"
OUT_SUMMARY_JSON = OUTPUT_DIR / "parsed_ocr_summary.json"
FAILED_JSONL = FAILED_DIR / "ocr_failures.jsonl"

MODEL_ID = os.getenv("PADDLEOCR_VL_MODEL", "PaddlePaddle/PaddleOCR-VL")
SERVED_MODEL_NAME = os.getenv("PADDLEOCR_VL_SERVED_MODEL_NAME", "PaddleOCR-VL-0.9B")
PORT = int(os.getenv("PADDLEOCR_VL_PORT", "8118"))

GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.90")
MAX_NUM_BATCHED_TOKENS = os.getenv("MAX_NUM_BATCHED_TOKENS", "32768")


# ============================================================
# vLLM server
# ============================================================

def vllm_base_url(with_v1: bool = True) -> str:
    base = f"http://127.0.0.1:{PORT}"
    return f"{base}/v1" if with_v1 else base


def check_vllm_ready(timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(f"{vllm_base_url(True)}/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_vllm(timeout_seconds: int = 900) -> None:
    started = time.time()

    while time.time() - started < timeout_seconds:
        if check_vllm_ready(timeout=5.0):
            return
        time.sleep(5)

    raise TimeoutError(f"vLLM server not ready after {timeout_seconds} seconds")


def start_vllm_server() -> subprocess.Popen:
    """
    Starts vLLM OpenAI-compatible server for PaddleOCR-VL.

    RTX 5090 32GB default:
      gpu-memory-utilization: 0.90
      max-num-batched-tokens: 32768

    If OOM:
      reduce MAX_NUM_BATCHED_TOKENS=16384
      reduce GPU_MEMORY_UTILIZATION=0.85
    """

    if check_vllm_ready():
        print(f"vLLM already running at {vllm_base_url(True)}")
        return None  # type: ignore[return-value]

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_ID,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        "0",
    ]

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    log_path = OUTPUT_DIR / "vllm_paddleocr_vl_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting vLLM server:")
    print(" ".join(cmd))
    print(f"vLLM log: {log_path}")

    log_file = log_path.open("a", encoding="utf-8")

    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
        text=True,
    )

    print("Waiting for vLLM server...")
    wait_for_vllm(timeout_seconds=900)
    print(f"vLLM ready: {vllm_base_url(True)}")

    return process


def stop_process(process: Optional[subprocess.Popen]) -> None:
    if not process:
        return

    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


# ============================================================
# PaddleOCR-VL client
# ============================================================

def create_paddleocr_vl_pipeline():
    from paddleocr import PaddleOCRVL

    # Official docs show both no-backend local mode and vLLM server backend.
    # We use vLLM because RTX 5090 should accelerate VL recognition.
    server_url_v1 = vllm_base_url(True)
    server_url_plain = vllm_base_url(False)

    try:
        return PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url=server_url_v1,
        )
    except Exception as e1:
        print(f"PaddleOCRVL init with /v1 failed: {e1}")
        print("Retrying without /v1...")
        return PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url=server_url_plain,
        )


def save_paddle_results(output: Any, raw_output_dir: Path) -> List[Dict[str, str]]:
    """
    Saves PaddleOCR-VL result objects to JSON and Markdown.

    Paddle result objects support:
      res.save_to_json(save_path=...)
      res.save_to_markdown(save_path=...)
    """
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for idx, res in enumerate(output):
        item_dir = raw_output_dir / f"res_{idx:04d}"
        item_dir.mkdir(parents=True, exist_ok=True)

        json_status = "not_saved"
        md_status = "not_saved"

        try:
            res.save_to_json(save_path=str(item_dir))
            json_status = "saved"
        except Exception as e:
            json_status = f"failed: {e}"

        try:
            res.save_to_markdown(save_path=str(item_dir))
            md_status = "saved"
        except Exception as e:
            md_status = f"failed: {e}"

        saved.append({
            "result_index": str(idx),
            "output_dir": item_dir.as_posix(),
            "json_status": json_status,
            "markdown_status": md_status,
        })

    return saved


# ============================================================
# Text extraction from Paddle outputs
# ============================================================

TEXT_KEYS = {
    "text",
    "content",
    "markdown",
    "html",
    "rec_text",
    "recognized_text",
    "label",
}


def collect_text_recursive(obj: Any, lines: List[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()

            if key in TEXT_KEYS and isinstance(v, str):
                txt = clean_text(v)
                if txt:
                    lines.append(txt)
            else:
                collect_text_recursive(v, lines)

    elif isinstance(obj, list):
        for item in obj:
            collect_text_recursive(item, lines)


def collect_raw_text_from_output_dir(raw_output_dir: Path) -> str:
    lines = []

    # Prefer markdown because PaddleOCR-VL formats reading order/table output there.
    for md_path in sorted(raw_output_dir.rglob("*.md")):
        try:
            txt = clean_text(md_path.read_text(encoding="utf-8", errors="replace"))
            if txt:
                lines.append(txt)
        except Exception:
            pass

    # Add JSON text fields if markdown is absent or incomplete.
    json_lines = []
    for json_path in sorted(raw_output_dir.rglob("*.json")):
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
            collect_text_recursive(obj, json_lines)
        except Exception:
            pass

    if len("\n".join(json_lines)) > len("\n".join(lines)):
        lines.extend(json_lines)

    # Deduplicate large repeated chunks.
    deduped = []
    seen = set()

    for line in lines:
        line = clean_text(line)
        if not line:
            continue

        key = line[:500]
        if key in seen:
            continue

        seen.add(key)
        deduped.append(line)

    return clean_text("\n\n".join(deduped))


# ============================================================
# OCR one document
# ============================================================

def run_ocr_one(pipeline: Any, queue_row: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    variant_id = queue_row["variant_id"]
    input_path = Path(queue_row["ocr_input_path"])

    if not input_path.exists():
        raise FileNotFoundError(f"OCR input file not found: {input_path}")

    variant_dir = PARSED_OCR_DIR / variant_id
    raw_output_dir = variant_dir / "paddle_raw"
    final_json_path = variant_dir / "parsed.json"

    if final_json_path.exists() and not force:
        return json.loads(final_json_path.read_text(encoding="utf-8"))

    if raw_output_dir.exists() and force:
        import shutil
        shutil.rmtree(raw_output_dir)

    variant_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    output = pipeline.predict(str(input_path))

    saved_outputs = save_paddle_results(output, raw_output_dir)
    raw_text = collect_raw_text_from_output_dir(raw_output_dir)

    detected_language = detect_language_from_text(raw_text)
    quality_flags = get_quality_flags(raw_text, detected_language)

    elapsed_seconds = round(time.time() - started, 3)

    record = {
        **queue_row,
        "detected_language": detected_language,
        "extraction_method": "paddleocr_vl_vllm",
        "raw_output_dir": raw_output_dir.as_posix(),
        "saved_outputs": saved_outputs,
        "char_count": len(raw_text),
        "quality_flags": quality_flags,
        "elapsed_seconds": elapsed_seconds,
        "raw_text": raw_text,
    }

    write_json(final_json_path, record)
    return record


def rebuild_parsed_ocr_jsonl() -> int:
    rows = []

    for p in sorted(PARSED_OCR_DIR.glob("*/parsed.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass

    write_jsonl(OUT_PARSED_OCR_JSONL, rows)
    return len(rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-start-server", action="store_true", help="Use already running vLLM server")
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument("--only-word-converted", action="store_true")
    parser.add_argument("--only-pdf-primary", action="store_true")
    args = parser.parse_args()

    ensure_base_dirs()
    PARSED_OCR_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(QUEUE_JSONL))

    if args.only_word_converted:
        rows = [r for r in rows if r.get("from_word_conversion") is True]

    if args.only_pdf_primary:
        rows = [r for r in rows if not r.get("from_word_conversion")]

    rows = rows[args.start:]

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    if not rows:
        print("No OCR queue rows found.")
        return

    print(f"OCR rows selected: {len(rows)}")
    print(f"Model: {MODEL_ID}")
    print(f"Served model name: {SERVED_MODEL_NAME}")
    print(f"Port: {PORT}")
    print(f"GPU_MEMORY_UTILIZATION: {GPU_MEMORY_UTILIZATION}")
    print(f"MAX_NUM_BATCHED_TOKENS: {MAX_NUM_BATCHED_TOKENS}")

    server_process = None

    try:
        if args.no_start_server:
            print("Using existing vLLM server...")
            wait_for_vllm(timeout_seconds=args.server_timeout)
        else:
            server_process = start_vllm_server()

        pipeline = create_paddleocr_vl_pipeline()

        parsed_ok = 0
        failed = 0
        skipped_existing = 0
        total_chars = 0
        total_seconds = 0.0

        for row in tqdm(rows, desc="PaddleOCR-VL"):
            final_json_path = PARSED_OCR_DIR / row["variant_id"] / "parsed.json"

            if final_json_path.exists() and not args.force:
                skipped_existing += 1
                continue

            try:
                result = run_ocr_one(pipeline, row, force=args.force)
                parsed_ok += 1
                total_chars += int(result.get("char_count") or 0)
                total_seconds += float(result.get("elapsed_seconds") or 0)

            except Exception as e:
                failed += 1
                append_jsonl(FAILED_JSONL, {
                    **row,
                    "ocr_status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        total_parsed_files = rebuild_parsed_ocr_jsonl()

        summary = {
            "ocr_rows_considered": len(rows),
            "parsed_ok": parsed_ok,
            "failed": failed,
            "skipped_existing": skipped_existing,
            "total_parsed_ocr_files": total_parsed_files,
            "total_chars_current_run": total_chars,
            "total_seconds_current_run": round(total_seconds, 3),
            "avg_seconds_per_success": round(total_seconds / parsed_ok, 3) if parsed_ok else None,
            "model_id": MODEL_ID,
            "served_model_name": SERVED_MODEL_NAME,
            "vllm_url": vllm_base_url(True),
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "outputs": {
                "parsed_ocr_jsonl": OUT_PARSED_OCR_JSONL.as_posix(),
                "parsed_ocr_dir": PARSED_OCR_DIR.as_posix(),
                "ocr_failures_jsonl": FAILED_JSONL.as_posix(),
                "summary_json": OUT_SUMMARY_JSON.as_posix(),
                "vllm_log": (OUTPUT_DIR / "vllm_paddleocr_vl_server.log").as_posix(),
            },
        }

        write_json(OUT_SUMMARY_JSON, summary)

        print("\nDONE")
        for k, v in summary.items():
            if k != "outputs":
                print(f"{k}: {v}")

        print("\nSaved:")
        for _, v in summary["outputs"].items():
            print(v)

    finally:
        # By default we keep server running only during this script.
        # For long sessions, use --no-start-server after manually starting vLLM.
        if server_process is not None:
            print("Stopping vLLM server...")
            stop_process(server_process)


if __name__ == "__main__":
    main()
