#!/usr/bin/env python3
import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Must be set before Docling / torch-heavy imports.
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("DOCLING_NUM_THREADS", "1"))
os.environ.setdefault("DOCLING_NUM_THREADS", os.getenv("DOCLING_NUM_THREADS", "1"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

from src.common import (
    FAILED_DIR,
    OCR_QUEUE_DIR,
    OUTPUT_DIR,
    PARSED_DOCLING_DIR,
    PARSED_OCR_DIR,
    PDF_WORK_DIR,
    PDF_VISUAL_ASSETS_DIR,
    PARSED_PDF_AGENTIC_DIR,
    PARSED_PDF_DOCLING_RAW_DIR,
    append_jsonl,
    clean_text,
    detect_language_from_text,
    ensure_base_dirs,
    find_icd_codes,
    get_quality_flags,
    iter_jsonl,
    stable_id,
    write_csv,
    write_json,
    write_jsonl,
)

QUEUE_JSONL = OCR_QUEUE_DIR / "ocr_queue.jsonl"

OUT_PARSED_DOCLING_JSONL = OUTPUT_DIR / "parsed_docling.jsonl"
OUT_PARSED_OCR_JSONL = OUTPUT_DIR / "parsed_ocr.jsonl"  # compatibility alias
OUT_VISUAL_MANIFEST_JSONL = OUTPUT_DIR / "pdf_visual_manifest.jsonl"
OUT_VISUAL_MANIFEST_CSV = OUTPUT_DIR / "pdf_visual_manifest.csv"
OUT_SUMMARY_JSON = OUTPUT_DIR / "parsed_docling_summary.json"
FAILED_JSONL = FAILED_DIR / "docling_surya_failures.jsonl"


# ============================================================
# Env config
# ============================================================

def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def get_ocr_langs() -> List[str]:
    raw = os.getenv("DOCLING_OCR_LANGS", "uz,ru,en")
    langs = [x.strip() for x in raw.split(",") if x.strip()]
    return langs or ["uz", "ru", "en"]


def get_batch_size() -> int:
    return max(1, env_int("PDF_BATCH_SIZE", 8))


def get_render_dpi() -> int:
    return max(96, env_int("PDF_RENDER_DPI", 220))


def get_crop_padding_px() -> int:
    return max(0, env_int("PDF_CROP_PADDING_PX", 12))


# ============================================================
# Docling converter
# ============================================================

def build_docling_pdf_converter():
    """
    PDF converter: Docling is the document orchestrator, SuryaOCR is the OCR
    backend via the docling-surya plugin.

    Gemma/VLM is intentionally not called here. Visual regions are cropped
    locally and exported to visual_manifest.jsonl for later enrichment.
    """

    from docling_surya import SuryaOcrOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()

    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = SuryaOcrOptions(
        lang=get_ocr_langs(),
        force_full_page_ocr=env_bool("DOCLING_FORCE_FULL_PAGE_OCR", True),
    )

    # Enabled by default in the agentic PDF pipeline; batching keeps memory sane.
    pipeline_options.do_table_structure = env_bool("DOCLING_DO_TABLE_STRUCTURE", True)

    # We render pages ourselves with PyMuPDF for predictable crop coordinates.
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.images_scale = env_float("DOCLING_IMAGES_SCALE", 1.0)

    # Required for docling-surya.
    pipeline_options.allow_external_plugins = True

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=env_int("DOCLING_NUM_THREADS", 1),
        device=AcceleratorDevice.AUTO,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


# ============================================================
# PDF/page helpers
# ============================================================

def split_pdf_to_batches(pdf_path: Path, work_dir: Path, batch_size: int) -> List[Dict[str, Any]]:
    """
    Create small page-batch PDFs so Docling never sees huge PDFs at once.
    This prevents std::bad_alloc and makes retry/recovery deterministic.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    src = fitz.open(str(pdf_path))
    batches = []

    try:
        page_count = len(src)
        for start in range(0, page_count, batch_size):
            end = min(start + batch_size, page_count)
            batch_no = len(batches) + 1
            batch_path = work_dir / f"batch_{batch_no:04d}_p{start + 1:04d}_{end:04d}.pdf"

            if not batch_path.exists():
                dst = fitz.open()
                dst.insert_pdf(src, from_page=start, to_page=end - 1)
                dst.save(str(batch_path), garbage=4, deflate=True)
                dst.close()

            batches.append({
                "batch_no": batch_no,
                "batch_path": batch_path,
                "global_start_page": start + 1,  # 1-indexed
                "global_end_page": end,
                "page_count": end - start,
            })
    finally:
        src.close()

    return batches


def collect_pdf_page_info(pdf_path: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    pages = []

    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            rect = page.rect
            pages.append({
                "page": i,
                "width_pts": float(rect.width),
                "height_pts": float(rect.height),
                "text_chars": len(text.strip()),
                "is_scanned_like": len(text.strip()) < 30,
            })
    finally:
        doc.close()

    return pages


def classify_document(page_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not page_info:
        return {"kind": "unknown", "scanned_page_ratio": 0.0}

    scanned = sum(1 for p in page_info if p.get("is_scanned_like"))
    ratio = scanned / max(len(page_info), 1)

    if ratio > 0.70:
        kind = "scanned"
    elif ratio > 0.10:
        kind = "mixed"
    else:
        kind = "digital"

    return {
        "kind": kind,
        "is_scanned": kind == "scanned",
        "is_mixed": kind == "mixed",
        "scanned_page_ratio": round(ratio, 3),
    }


def render_pages(pdf_path: Path, pages_dir: Path, dpi: int, keep_existing: bool = True) -> Dict[int, Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    out = {}

    try:
        for idx, page in enumerate(doc, start=1):
            page_path = pages_dir / f"page_{idx:04d}.png"
            out[idx] = page_path

            if keep_existing and page_path.exists():
                continue

            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
            pix.save(str(page_path))
    finally:
        doc.close()

    return out


def get_pymupdf_page_text(pdf_path: Path, page_no: int) -> str:
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_no - 1]
        return clean_text(page.get_text("text") or "")
    finally:
        doc.close()


# ============================================================
# Region/block extraction
# ============================================================

def safe_label(item: Any, default: str) -> str:
    label = getattr(item, "label", default)
    if hasattr(label, "value"):
        return str(label.value)
    return str(label or default)


def clamp01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def normalize_docling_bbox(bb: Any, page_w: float, page_h: float) -> Dict[str, float]:
    x0 = clamp01(float(bb.l) / page_w)
    x1 = clamp01(float(bb.r) / page_w)

    # Docling provenance bbox normally uses PDF bottom-left origin.
    y0 = clamp01(1.0 - (float(bb.t) / page_h))
    y1 = clamp01(1.0 - (float(bb.b) / page_h))

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def bbox_area(b: Dict[str, float]) -> float:
    return max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])


def crop_region(
    page_image_path: Path,
    bbox: Dict[str, float],
    out_path: Path,
    pad_px: int,
) -> Optional[Path]:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(page_image_path).convert("RGB")
        w, h = img.size

        left = max(0, int(bbox["x0"] * w) - pad_px)
        top = max(0, int(bbox["y0"] * h) - pad_px)
        right = min(w, int(bbox["x1"] * w) + pad_px)
        bottom = min(h, int(bbox["y1"] * h) + pad_px)

        if right <= left or bottom <= top:
            return None

        crop = img.crop((left, top, right, bottom))
        crop.save(out_path)
        return out_path
    except Exception:
        return None


def table_markdown_is_good(md: str) -> Tuple[bool, List[str]]:
    reasons = []

    if not md or len(md.strip()) < 20:
        return False, ["empty_or_short_table_markdown"]

    lines = [x.strip() for x in md.splitlines() if x.strip()]
    pipe_lines = [x for x in lines if "|" in x]

    if len(pipe_lines) < 2:
        return False, ["not_enough_markdown_table_rows"]

    col_counts = []
    for line in pipe_lines:
        parts = [p.strip() for p in line.strip("|").split("|")]
        if not set("".join(parts)) <= {"-", ":", " "}:
            col_counts.append(len(parts))

    if not col_counts:
        return False, ["no_usable_table_cells"]

    most_common = max(set(col_counts), key=col_counts.count)
    consistency = col_counts.count(most_common) / len(col_counts)

    if most_common < 2:
        reasons.append("table_has_fewer_than_2_columns")

    if consistency < 0.75:
        reasons.append("inconsistent_row_column_counts")

    return len(reasons) == 0, reasons


def infer_visual_type(label: str, context: str = "") -> str:
    s = f"{label} {context}".lower()

    if any(k in s for k in ["flow", "algorithm", "алгоритм", "схема", "блок-схема"]):
        return "flowchart"

    if any(k in s for k in ["chart", "graph", "диаграм", "график", "шкала", "scale"]):
        return "chart"

    if any(k in s for k in ["table", "таблица", "жадвал"]):
        return "complex_table"

    return "figure"


def block_text_for_context(block: Dict[str, Any]) -> str:
    return clean_text(block.get("text") or block.get("markdown") or "")


def nearby_context(blocks: List[Dict[str, Any]], order: int, page: int, window: int = 2) -> Dict[str, str]:
    same_page = [b for b in blocks if b.get("page") == page and b.get("type") in {"text", "table"}]
    before = [b for b in same_page if int(b.get("order", 0)) < order]
    after = [b for b in same_page if int(b.get("order", 0)) > order]

    before_text = "\n".join(block_text_for_context(b) for b in before[-window:])
    after_text = "\n".join(block_text_for_context(b) for b in after[:window])

    caption = ""
    for candidate in before[-1:] + after[:1]:
        label = str(candidate.get("docling_label", "")).lower()
        text = block_text_for_context(candidate)
        if "caption" in label or text.lower().startswith(("figure", "рис", "расм", "таблица", "жадвал")):
            caption = text
            break

    return {
        "before_text": clean_text(before_text)[:2000],
        "after_text": clean_text(after_text)[:2000],
        "caption": clean_text(caption)[:1000],
    }


def extract_blocks_from_docling(
    doc: Any,
    batch: Dict[str, Any],
    original_page_info: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    blocks = []
    local_order = 0

    page_sizes = {
        int(p["page"]): {
            "w": float(p["width_pts"]),
            "h": float(p["height_pts"]),
        }
        for p in original_page_info
    }

    def add_item(item: Any, typ: str, text: str = "", markdown: str = "", label: str = ""):
        nonlocal local_order
        provs = getattr(item, "prov", []) or []

        for prov in provs:
            local_page = int(prov.page_no)
            global_page = int(batch["global_start_page"]) + local_page - 1

            if global_page not in page_sizes:
                continue

            ps = page_sizes[global_page]

            try:
                bbox = normalize_docling_bbox(prov.bbox, ps["w"], ps["h"])
            except Exception:
                continue

            if bbox_area(bbox) <= 0.00001:
                continue

            blocks.append({
                "id": f"p{global_page:04d}_b{local_order:06d}",
                "page": global_page,
                "order": local_order,
                "type": typ,
                "bbox": bbox,
                "text": clean_text(text),
                "markdown": markdown or clean_text(text),
                "docling_label": label or typ,
                "source": {
                    "engine": "docling_surya",
                    "batch_no": batch["batch_no"],
                    "local_page": local_page,
                },
                "quality": {
                    "needs_review": False,
                    "reasons": [],
                },
            })
            local_order += 1

    for item in getattr(doc, "texts", []) or []:
        add_item(
            item,
            typ="text",
            text=getattr(item, "text", "") or "",
            markdown=getattr(item, "text", "") or "",
            label=safe_label(item, "text"),
        )

    for item in getattr(doc, "tables", []) or []:
        try:
            md = item.export_to_markdown()
        except Exception:
            md = ""
        good, reasons = table_markdown_is_good(md)
        add_item(
            item,
            typ="table",
            text=clean_text(md),
            markdown=md,
            label="table",
        )
        if blocks:
            blocks[-1]["quality"]["needs_review"] = not good
            blocks[-1]["quality"]["reasons"] = reasons

    for item in getattr(doc, "pictures", []) or []:
        add_item(
            item,
            typ="pending_visual",
            text="",
            markdown="",
            label=safe_label(item, "figure"),
        )

    blocks.sort(key=lambda b: (int(b["page"]), float(b["bbox"]["y0"]), float(b["bbox"]["x0"]), int(b["order"])))

    # Reassign global order across the batch.
    for i, b in enumerate(blocks):
        b["order"] = i
        b["id"] = f"p{int(b['page']):04d}_b{i:06d}"

    return blocks


def add_page_fallback_blocks(
    blocks: List[Dict[str, Any]],
    pdf_path: Path,
    page_info: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Coverage safety net. If Docling returns no block for a page:
      - digital page: keep PyMuPDF text fallback
      - scanned/empty page: export full-page pending_visual block
    """
    pages_with_blocks = {int(b["page"]) for b in blocks}
    next_order = len(blocks)

    for p in page_info:
        page_no = int(p["page"])
        if page_no in pages_with_blocks:
            continue

        text = get_pymupdf_page_text(pdf_path, page_no)

        if text:
            blocks.append({
                "id": f"p{page_no:04d}_b{next_order:06d}",
                "page": page_no,
                "order": next_order,
                "type": "text",
                "bbox": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
                "text": text,
                "markdown": text,
                "docling_label": "pymupdf_page_text_fallback",
                "source": {"engine": "pymupdf", "fallback": True},
                "quality": {"needs_review": True, "reasons": ["docling_page_missing_used_pymupdf_text_fallback"]},
            })
        else:
            blocks.append({
                "id": f"p{page_no:04d}_b{next_order:06d}",
                "page": page_no,
                "order": next_order,
                "type": "pending_visual",
                "bbox": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
                "text": "",
                "markdown": "",
                "docling_label": "full_page_missing_docling_fallback",
                "source": {"engine": "pipeline", "fallback": True},
                "quality": {"needs_review": True, "reasons": ["page_missing_docling_blocks_exported_for_later_vlm"]},
            })

        next_order += 1

    blocks.sort(key=lambda b: (int(b["page"]), int(b["order"])))
    for i, b in enumerate(blocks):
        b["order"] = i
        b["id"] = f"p{int(b['page']):04d}_b{i:06d}"

    return blocks


def export_visual_assets(
    variant_id: str,
    blocks: List[Dict[str, Any]],
    page_images: Dict[int, Path],
    assets_dir: Path,
    pad_px: int,
) -> List[Dict[str, Any]]:
    crops_dir = assets_dir / variant_id / "crops"
    manifest_path = assets_dir / variant_id / "visual_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    # Tables are also cropped when Docling table is weak/complex.
    visual_indexes = []
    for idx, b in enumerate(blocks):
        if b.get("type") == "pending_visual":
            visual_indexes.append(idx)
        elif b.get("type") == "table" and b.get("quality", {}).get("needs_review"):
            visual_indexes.append(idx)

    for idx in visual_indexes:
        b = blocks[idx]
        page_no = int(b["page"])
        page_img = page_images.get(page_no)
        if not page_img or not page_img.exists():
            b.setdefault("quality", {}).setdefault("reasons", []).append("page_image_missing_for_visual_crop")
            b["quality"]["needs_review"] = True
            continue

        context = nearby_context(blocks, int(b["order"]), page_no)
        suggested_type = infer_visual_type(b.get("docling_label", ""), " ".join(context.values()))

        asset_id = f"{variant_id}_p{page_no:04d}_b{int(b['order']):06d}"
        crop_path = crops_dir / f"{asset_id}.png"

        saved_crop = crop_region(
            page_image_path=page_img,
            bbox=b["bbox"],
            out_path=crop_path,
            pad_px=pad_px,
        )

        if not saved_crop:
            b.setdefault("quality", {}).setdefault("reasons", []).append("visual_crop_failed")
            b["quality"]["needs_review"] = True
            continue

        asset = {
            "doc_id": variant_id,
            "variant_id": variant_id,
            "asset_id": asset_id,
            "page": page_no,
            "order": int(b["order"]),
            "block_id": b["id"],
            "suggested_type": suggested_type if b.get("type") == "pending_visual" else "complex_table",
            "bbox": b["bbox"],
            "crop_path": saved_crop.as_posix(),
            "caption": context.get("caption", ""),
            "before_text": context.get("before_text", ""),
            "after_text": context.get("after_text", ""),
            "status": "pending_vlm",
        }

        b["type"] = "pending_visual" if b.get("type") == "pending_visual" else "table"
        b["visual_asset"] = {
            "asset_id": asset_id,
            "crop_path": saved_crop.as_posix(),
            "nearby_context": context,
            "suggested_type": asset["suggested_type"],
        }
        b["visual_extraction"] = None
        b.setdefault("quality", {})["needs_vlm"] = True
        b["quality"]["needs_review"] = True
        reasons = b["quality"].setdefault("reasons", [])
        reasons.append("visual/table crop exported for later VLM enrichment")
        b["quality"]["reasons"] = sorted(set(reasons))

        manifest_rows.append(asset)

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return manifest_rows


def group_pages(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for b in sorted(blocks, key=lambda x: (int(x["page"]), int(x["order"]))):
        by_page.setdefault(int(b["page"]), []).append(b)

    return [{"page": page, "blocks": rows} for page, rows in sorted(by_page.items())]


def build_raw_text_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    parts = []
    for b in sorted(blocks, key=lambda x: (int(x["page"]), int(x["order"]))):
        if b.get("type") in {"text", "table"}:
            t = b.get("markdown") or b.get("text") or ""
            if t.strip():
                parts.append(t)
    return clean_text("\n\n".join(parts))


def build_markdown_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    parts = []
    current_page = None
    for b in sorted(blocks, key=lambda x: (int(x["page"]), int(x["order"]))):
        page = int(b["page"])
        if page != current_page:
            current_page = page
            parts.append(f"\n\n<!-- page {page} -->\n")

        if b.get("type") in {"text", "table"}:
            t = b.get("markdown") or b.get("text") or ""
            if t.strip():
                parts.append(t.strip())
        elif b.get("type") == "pending_visual":
            asset = b.get("visual_asset") or {}
            parts.append(f"[PENDING_VISUAL asset_id={asset.get('asset_id', '')} type={asset.get('suggested_type', '')}]")

    return clean_text("\n\n".join(parts))


# ============================================================
# Main extraction
# ============================================================

def run_docling_surya_one(
    converter: Any,
    queue_row: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    variant_id = queue_row["variant_id"]
    input_path = Path(queue_row["ocr_input_path"])

    if not input_path.exists():
        raise FileNotFoundError(f"Docling input PDF not found: {input_path}")

    compatibility_dir = PARSED_DOCLING_DIR / variant_id
    final_json_path = compatibility_dir / "parsed.json"

    agentic_dir = PARSED_PDF_AGENTIC_DIR / variant_id
    agentic_json_path = agentic_dir / "agentic.json"

    raw_root = PARSED_PDF_DOCLING_RAW_DIR / variant_id
    work_dir = PDF_WORK_DIR / variant_id
    assets_dir = PDF_VISUAL_ASSETS_DIR

    if final_json_path.exists() and agentic_json_path.exists() and not force:
        return json.loads(final_json_path.read_text(encoding="utf-8"))

    if force:
        for d in [compatibility_dir, agentic_dir, raw_root, work_dir, assets_dir / variant_id]:
            if d.exists():
                shutil.rmtree(d)

    compatibility_dir.mkdir(parents=True, exist_ok=True)
    agentic_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    page_info = collect_pdf_page_info(input_path)
    doc_class = classify_document(page_info)
    page_images = render_pages(
        input_path,
        assets_dir / variant_id / "pages",
        dpi=get_render_dpi(),
        keep_existing=True,
    )

    batches = split_pdf_to_batches(input_path, work_dir / "batches", get_batch_size())

    all_blocks: List[Dict[str, Any]] = []
    batch_outputs: List[Dict[str, Any]] = []
    errors: List[str] = []

    for batch in batches:
        batch_dir = raw_root / f"batch_{batch['batch_no']:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = converter.convert(str(batch["batch_path"]))
            doc = result.document

            md = doc.export_to_markdown()
            docling_dict = doc.export_to_dict()

            (batch_dir / "document.md").write_text(md, encoding="utf-8")
            (batch_dir / "document.docling.json").write_text(
                json.dumps(docling_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            blocks = extract_blocks_from_docling(doc, batch, page_info)
            all_blocks.extend(blocks)

            batch_outputs.append({
                "batch_no": batch["batch_no"],
                "batch_path": batch["batch_path"].as_posix(),
                "global_start_page": batch["global_start_page"],
                "global_end_page": batch["global_end_page"],
                "markdown_path": (batch_dir / "document.md").as_posix(),
                "docling_json_path": (batch_dir / "document.docling.json").as_posix(),
                "counts": {
                    "texts": len(getattr(doc, "texts", []) or []),
                    "tables": len(getattr(doc, "tables", []) or []),
                    "pictures": len(getattr(doc, "pictures", []) or []),
                },
                "status": "ok",
            })

        except Exception as exc:
            msg = f"batch {batch['batch_no']} pages {batch['global_start_page']}-{batch['global_end_page']} failed: {exc}"
            errors.append(msg)
            batch_outputs.append({
                "batch_no": batch["batch_no"],
                "batch_path": batch["batch_path"].as_posix(),
                "global_start_page": batch["global_start_page"],
                "global_end_page": batch["global_end_page"],
                "status": "failed",
                "error": str(exc),
            })

    all_blocks = add_page_fallback_blocks(all_blocks, input_path, page_info)

    visual_rows = export_visual_assets(
        variant_id=variant_id,
        blocks=all_blocks,
        page_images=page_images,
        assets_dir=assets_dir,
        pad_px=get_crop_padding_px(),
    )

    raw_text = build_raw_text_from_blocks(all_blocks)
    merged_markdown = build_markdown_from_blocks(all_blocks)

    detected_language = detect_language_from_text(raw_text)
    quality_flags = get_quality_flags(raw_text, detected_language)

    if errors:
        quality_flags.append("docling_batch_errors")

    if visual_rows:
        quality_flags.append("pending_visual_assets")

    if len(raw_text) < env_int("PDF_MIN_CHARS", 500):
        quality_flags.append("pdf_low_char_count")

    elapsed_seconds = round(time.time() - started, 3)

    md_path = compatibility_dir / "document.md"
    parsed_json_path = compatibility_dir / "parsed.json"
    md_path.write_text(merged_markdown, encoding="utf-8")

    visual_manifest_path = assets_dir / variant_id / "visual_manifest.jsonl"

    agentic_output = {
        "variant_id": variant_id,
        "source_path": input_path.as_posix(),
        "classification": doc_class,
        "processing": {
            "pipeline": "agentic_pdf_docling_surya_deferred_vlm",
            "defer_vlm": True,
            "batch_size": get_batch_size(),
            "render_dpi": get_render_dpi(),
            "total_pages": len(page_info),
            "total_blocks": len(all_blocks),
            "pending_visual_count": len(visual_rows),
            "errors": errors,
        },
        "batches": batch_outputs,
        "pages": group_pages(all_blocks),
    }

    agentic_json_path.parent.mkdir(parents=True, exist_ok=True)
    agentic_json_path.write_text(json.dumps(agentic_output, ensure_ascii=False, indent=2), encoding="utf-8")

    record = {
        **queue_row,
        "detected_language": detected_language,
        "extraction_method": "agentic_pdf_docling_surya_deferred_vlm",
        "raw_output_dir": compatibility_dir.as_posix(),
        "agentic_json_path": agentic_json_path.as_posix(),
        "markdown_path": md_path.as_posix(),
        "docling_json_path": "",  # batch raw JSON paths are in raw_docling_outputs.
        "raw_docling_dir": raw_root.as_posix(),
        "visual_manifest_path": visual_manifest_path.as_posix(),
        "visual_assets_dir": (assets_dir / variant_id).as_posix(),
        "pending_visual_count": len(visual_rows),
        "visual_asset_count": len(visual_rows),
        "docling_batch_count": len(batches),
        "docling_batch_failed_count": sum(1 for b in batch_outputs if b.get("status") != "ok"),
        "page_count": len(page_info),
        "char_count": len(raw_text),
        "quality_flags": sorted(set(quality_flags)),
        "elapsed_seconds": elapsed_seconds,
        "raw_text": raw_text,
        "icd_codes_detected": find_icd_codes(raw_text),
        "saved_outputs": {
            "agentic_json_path": agentic_json_path.as_posix(),
            "markdown_path": md_path.as_posix(),
            "raw_docling_dir": raw_root.as_posix(),
            "visual_manifest_path": visual_manifest_path.as_posix(),
        },
    }

    write_json(parsed_json_path, record)

    if not env_bool("PDF_KEEP_PAGE_IMAGES", False):
        # Keep crops + manifest, remove rendered full page images to save space.
        shutil.rmtree(assets_dir / variant_id / "pages", ignore_errors=True)

    if not env_bool("PDF_KEEP_WORK_DIR", False):
        shutil.rmtree(work_dir, ignore_errors=True)

    return record


def rebuild_parsed_docling_jsonl() -> Tuple[int, int]:
    rows = []
    visual_rows = []

    for p in sorted(PARSED_DOCLING_DIR.glob("*/parsed.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            rows.append(row)

            manifest_path = Path(row.get("visual_manifest_path") or "")
            if manifest_path.exists():
                for item in iter_jsonl(manifest_path):
                    visual_rows.append(item)

        except Exception:
            pass

    write_jsonl(OUT_PARSED_DOCLING_JSONL, rows)
    write_jsonl(OUT_PARSED_OCR_JSONL, rows)
    write_jsonl(OUT_VISUAL_MANIFEST_JSONL, visual_rows)

    if visual_rows:
        write_csv(
            OUT_VISUAL_MANIFEST_CSV,
            visual_rows,
            fieldnames=[
                "variant_id",
                "asset_id",
                "page",
                "order",
                "block_id",
                "suggested_type",
                "crop_path",
                "status",
                "caption",
            ],
        )

    return len(rows), len(visual_rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-backfill", action="store_true")
    parser.add_argument("--only-primary", action="store_true")
    args = parser.parse_args()

    ensure_base_dirs()
    PARSED_DOCLING_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_OCR_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_PDF_AGENTIC_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_PDF_DOCLING_RAW_DIR.mkdir(parents=True, exist_ok=True)
    PDF_VISUAL_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(QUEUE_JSONL))

    if args.only_backfill:
        rows = [r for r in rows if str(r.get("ocr_reason", "")).startswith("backfill")]

    if args.only_primary:
        rows = [r for r in rows if not str(r.get("ocr_reason", "")).startswith("backfill")]

    rows = rows[args.start:]

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    if not rows:
        print("No Docling OCR queue rows found.")
        return

    print(f"Agentic Docling+Surya PDF rows selected: {len(rows)}")
    print(f"OCR langs: {get_ocr_langs()}")
    print(f"Batch size: {get_batch_size()}")
    print(f"Render DPI: {get_render_dpi()}")
    print(f"Table structure: {env_bool('DOCLING_DO_TABLE_STRUCTURE', True)}")
    print(f"Docling threads: {env_int('DOCLING_NUM_THREADS', 1)}")
    print(f"Surya backend: {os.getenv('SURYA_INFERENCE_BACKEND', 'vllm')}")
    print(f"Defer VLM: true")

    converter = build_docling_pdf_converter()

    parsed_ok = 0
    failed = 0
    skipped_existing = 0
    total_chars = 0
    total_seconds = 0.0
    total_visual_assets = 0

    for row in tqdm(rows, desc="AgenticPDF"):
        final_json_path = PARSED_DOCLING_DIR / row["variant_id"] / "parsed.json"
        agentic_json_path = PARSED_PDF_AGENTIC_DIR / row["variant_id"] / "agentic.json"

        if final_json_path.exists() and agentic_json_path.exists() and not args.force:
            skipped_existing += 1
            continue

        try:
            result = run_docling_surya_one(converter, row, force=args.force)

            parsed_ok += 1
            total_chars += int(result.get("char_count") or 0)
            total_seconds += float(result.get("elapsed_seconds") or 0)
            total_visual_assets += int(result.get("visual_asset_count") or 0)

        except Exception as e:
            failed += 1

            append_jsonl(
                FAILED_JSONL,
                {
                    **row,
                    "ocr_status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    total_parsed_files, total_manifest_assets = rebuild_parsed_docling_jsonl()

    summary = {
        "ocr_rows_considered": len(rows),
        "parsed_ok": parsed_ok,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "total_parsed_docling_files": total_parsed_files,
        "total_visual_assets_manifest": total_manifest_assets,
        "total_chars_current_run": total_chars,
        "total_seconds_current_run": round(total_seconds, 3),
        "avg_seconds_per_success": round(total_seconds / parsed_ok, 3) if parsed_ok else None,
        "ocr_engine": "docling_surya",
        "pipeline": "agentic_pdf_docling_surya_deferred_vlm",
        "ocr_langs": get_ocr_langs(),
        "batch_size": get_batch_size(),
        "render_dpi": get_render_dpi(),
        "defer_vlm": True,
        "do_table_structure": env_bool("DOCLING_DO_TABLE_STRUCTURE", True),
        "docling_num_threads": env_int("DOCLING_NUM_THREADS", 1),
        "surya_backend": os.getenv("SURYA_INFERENCE_BACKEND", "vllm"),
    }

    write_json(OUT_SUMMARY_JSON, summary)

    print("\nDONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
