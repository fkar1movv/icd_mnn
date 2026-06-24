#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.common import (
    FAILED_DIR,
    LLM_CLEANED_DIR,
    OUTPUT_DIR,
    append_jsonl,
    clean_text,
    ensure_base_dirs,
    find_icd_codes,
    iter_jsonl,
    write_json,
    write_jsonl,
)


INPUT_RAW_JSONL = OUTPUT_DIR / "parsed_guidelines_raw.jsonl"
OUT_CLEANED_JSONL = OUTPUT_DIR / "parsed_guidelines_cleaned.jsonl"
OUT_SUMMARY_JSON = OUTPUT_DIR / "llm_cleanup_summary.json"
FAILED_JSONL = FAILED_DIR / "llm_cleanup_failures.jsonl"

GEMMA_API_URL = os.getenv("GEMMA_API_URL", "").strip()
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY", "").strip()
USE_LLM_CLEANUP = os.getenv("USE_LLM_CLEANUP", "false").lower() == "true"

DEFAULT_MODEL = os.getenv("GEMMA_MODEL", "gemma-31b")


SYSTEM_PROMPT = """
You clean OCR/document text formatting only.

Allowed:
- fix broken line breaks
- remove repeated headers/footers
- normalize bullets and numbering
- join words broken by line wrapping
- preserve tables as readable markdown-like text
- preserve section headings

Strictly forbidden:
- do not add medical facts
- do not remove drug names
- do not add drug names
- do not infer ICD/MKB codes
- do not translate
- do not summarize
- do not change clinical meaning

Return only valid JSON:
{
  "clean_text": "...",
  "notes": ["..."]
}
""".strip()


def split_text(text: str, max_chars: int = 10000, overlap: int = 300) -> List[str]:
    text = clean_text(text)

    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        # Prefer paragraph boundary.
        boundary = text.rfind("\n\n", start, end)
        if boundary > start + max_chars * 0.55:
            end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def extract_json_from_response(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract JSON object from messy response.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM response")

    return json.loads(m.group(0))


def validate_cleaned_chunk(raw_chunk: str, cleaned_chunk: str) -> List[str]:
    flags = []

    raw_chunk = clean_text(raw_chunk)
    cleaned_chunk = clean_text(cleaned_chunk)

    if not cleaned_chunk:
        flags.append("empty_cleaned_chunk")
        return flags

    raw_len = max(len(raw_chunk), 1)
    clean_len = len(cleaned_chunk)
    ratio = clean_len / raw_len

    if ratio < 0.55:
        flags.append("cleaned_too_short")

    if ratio > 1.45:
        flags.append("cleaned_too_long")

    raw_icd = set(find_icd_codes(raw_chunk))
    clean_icd = set(find_icd_codes(cleaned_chunk))

    missing_icd = sorted(raw_icd - clean_icd)
    if missing_icd:
        flags.append("icd_codes_removed:" + ",".join(missing_icd[:20]))

    return flags


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def call_gemma_cleanup(chunk: str, model: str, timeout: int = 180) -> Dict[str, Any]:
    if not GEMMA_API_URL:
        raise RuntimeError("GEMMA_API_URL is empty")

    headers = {
        "Content-Type": "application/json",
    }

    if GEMMA_API_KEY:
        headers["Authorization"] = f"Bearer {GEMMA_API_KEY}"

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Clean this document text formatting only. "
                    "Return JSON only.\n\n"
                    f"TEXT:\n{chunk}"
                ),
            },
        ],
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(GEMMA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    # OpenAI-compatible format.
    if "choices" in data:
        content = data["choices"][0]["message"]["content"]
        return extract_json_from_response(content)

    # Simple custom API fallback.
    if "content" in data:
        return extract_json_from_response(data["content"])

    if "text" in data:
        return extract_json_from_response(data["text"])

    return data


def deterministic_cleanup_only(raw_text: str) -> str:
    """
    Safe non-LLM cleanup.
    Does not rewrite medical content.
    """
    text = clean_text(raw_text)

    # Remove excessive page separators if OCR created many.
    text = re.sub(r"\n\s*[-–—]{3,}\s*\n", "\n\n", text)

    # Normalize repeated spaces around punctuation.
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    # Normalize bullet spacing.
    text = re.sub(r"\n\s*[•●]\s*", "\n- ", text)

    return clean_text(text)


def cleanup_record(row: Dict[str, Any], args) -> Dict[str, Any]:
    variant_id = row["variant_id"]
    raw_text = row.get("raw_text") or ""

    out_path = LLM_CLEANED_DIR / f"{variant_id}.json"

    if out_path.exists() and not args.force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    raw_text = clean_text(raw_text)

    if not raw_text:
        result = {
            **row,
            "clean_text": "",
            "cleanup_method": "empty_raw_text",
            "cleanup_flags": ["empty_raw_text"],
            "llm_cleanup_used": False,
        }
        write_json(out_path, result)
        return result

    # If disabled, do deterministic cleanup and copy.
    if not USE_LLM_CLEANUP and not args.use_llm:
        clean = deterministic_cleanup_only(raw_text)
        result = {
            **row,
            "clean_text": clean,
            "cleanup_method": "deterministic_cleanup_only",
            "cleanup_flags": [],
            "llm_cleanup_used": False,
        }
        write_json(out_path, result)
        return result

    chunks = split_text(raw_text, max_chars=args.max_chunk_chars, overlap=args.overlap_chars)

    cleaned_chunks = []
    chunk_reports = []
    any_rejected = False

    for idx, chunk in enumerate(chunks):
        try:
            llm_result = call_gemma_cleanup(chunk, model=args.model, timeout=args.timeout)
            cleaned_chunk = clean_text(llm_result.get("clean_text") or "")

            validation_flags = validate_cleaned_chunk(chunk, cleaned_chunk)

            if validation_flags:
                any_rejected = True
                cleaned_chunks.append(chunk)
                chunk_reports.append({
                    "chunk_index": idx,
                    "status": "rejected_used_raw_chunk",
                    "validation_flags": validation_flags,
                })
            else:
                cleaned_chunks.append(cleaned_chunk)
                chunk_reports.append({
                    "chunk_index": idx,
                    "status": "accepted",
                    "validation_flags": [],
                })

        except Exception as e:
            any_rejected = True
            cleaned_chunks.append(chunk)
            chunk_reports.append({
                "chunk_index": idx,
                "status": "failed_used_raw_chunk",
                "error": str(e),
            })

    clean = clean_text("\n\n".join(cleaned_chunks))

    final_validation_flags = validate_cleaned_chunk(raw_text, clean)

    # Final safety fallback: if full text validation fails badly, preserve raw.
    cleanup_flags = []
    if any_rejected:
        cleanup_flags.append("some_chunks_rejected_or_failed")

    if final_validation_flags:
        cleanup_flags.extend(["final_validation:" + f for f in final_validation_flags])
        if any(
            f.startswith("icd_codes_removed")
            or f == "cleaned_too_short"
            for f in final_validation_flags
        ):
            clean = deterministic_cleanup_only(raw_text)
            cleanup_flags.append("final_fallback_to_deterministic_cleanup")

    result = {
        **row,
        "clean_text": clean,
        "cleanup_method": "gemma_light_formatting_cleanup",
        "cleanup_flags": cleanup_flags,
        "llm_cleanup_used": True,
        "cleanup_chunk_count": len(chunks),
        "cleanup_chunk_reports": chunk_reports,
    }

    write_json(out_path, result)
    return result


def safe_worker(row: Dict[str, Any], args) -> Dict[str, Any]:
    try:
        return cleanup_record(row, args)
    except Exception as e:
        fail = {
            "variant_id": row.get("variant_id"),
            "source_path": row.get("source_path"),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        append_jsonl(FAILED_JSONL, fail)

        # Never lose data: return raw as clean.
        raw_text = clean_text(row.get("raw_text") or "")
        return {
            **row,
            "clean_text": deterministic_cleanup_only(raw_text),
            "cleanup_method": "cleanup_failed_deterministic_fallback",
            "cleanup_flags": ["cleanup_failed"],
            "llm_cleanup_used": False,
        }


def rebuild_cleaned_jsonl() -> int:
    rows = []

    for p in sorted(LLM_CLEANED_DIR.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass

    write_jsonl(OUT_CLEANED_JSONL, rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-chunk-chars", type=int, default=10000)
    parser.add_argument("--overlap-chars", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    ensure_base_dirs()
    LLM_CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(INPUT_RAW_JSONL))

    rows = rows[args.start:]

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    if not rows:
        raise FileNotFoundError(f"No rows found in {INPUT_RAW_JSONL}")

    print(f"Rows selected: {len(rows)}")
    print(f"USE_LLM_CLEANUP env: {USE_LLM_CLEANUP}")
    print(f"--use-llm arg: {args.use_llm}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")

    cleaned = []
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(safe_worker, row, args) for row in rows]

        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM cleanup"):
            try:
                cleaned.append(future.result())
            except Exception:
                failed += 1

    total_cleaned_files = rebuild_cleaned_jsonl()

    summary = {
        "rows_considered": len(rows),
        "cleaned_current_run": len(cleaned),
        "failed_current_run": failed,
        "total_cleaned_files": total_cleaned_files,
        "llm_enabled": USE_LLM_CLEANUP or args.use_llm,
        "model": args.model,
        "outputs": {
            "cleaned_jsonl": OUT_CLEANED_JSONL.as_posix(),
            "cleaned_dir": LLM_CLEANED_DIR.as_posix(),
            "failures_jsonl": FAILED_JSONL.as_posix(),
            "summary_json": OUT_SUMMARY_JSON.as_posix(),
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


if __name__ == "__main__":
    main()
