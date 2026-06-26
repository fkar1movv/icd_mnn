import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from ftfy import fix_text
except Exception:
    fix_text = None


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_GUIDELINES_DIR = Path(os.getenv("RAW_GUIDELINES_DIR", "data/raw_guidelines"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/outputs"))
FAILED_DIR = Path(os.getenv("FAILED_DIR", "data/failed"))

PARSED_WORD_DIR = Path(os.getenv("PARSED_WORD_DIR", "data/parsed_word"))

# New Docling + Surya output directory.
# PARSED_OCR_DIR is kept as a compatibility alias because older merge scripts
# still import PARSED_OCR_DIR.
PARSED_DOCLING_DIR = Path(os.getenv("PARSED_DOCLING_DIR", "data/parsed_docling"))
PARSED_OCR_DIR = Path(os.getenv("PARSED_OCR_DIR", str(PARSED_DOCLING_DIR)))

CONVERTED_DIR = Path(os.getenv("CONVERTED_DIR", "data/converted"))
OCR_QUEUE_DIR = Path(os.getenv("OCR_QUEUE_DIR", "data/ocr_queue"))
LLM_CLEANED_DIR = Path(os.getenv("LLM_CLEANED_DIR", "data/llm_cleaned"))
LOG_DIR = Path(os.getenv("LOG_DIR", "data/logs"))

LIBREOFFICE_CMD = os.getenv("LIBREOFFICE_CMD", "soffice").strip('"').strip("'")

SUPPORTED_EXTENSIONS = {".docx", ".doc", ".pdf"}

WORD_EXTENSIONS = {".docx", ".doc"}
PDF_EXTENSIONS = {".pdf"}

REQUEST_LANGUAGES = ["ru", "uz_cyr", "uz_lat"]
FALLBACK_LANGUAGE_PRIORITY = ["ru", "uz_cyr", "uz_lat", "unknown"]


def ensure_base_dirs() -> None:
    for p in [
        RAW_GUIDELINES_DIR,
        OUTPUT_DIR,
        FAILED_DIR,
        PARSED_WORD_DIR,
        PARSED_DOCLING_DIR,
        PARSED_OCR_DIR,
        CONVERTED_DIR,
        OCR_QUEUE_DIR,
        LLM_CLEANED_DIR,
        LOG_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


# ============================================================
# JSON / JSONL / CSV helpers
# ============================================================

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    tmp.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path):
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        keys = []
        seen = set()

        for row in rows:
            for k in row.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)

        fieldnames = keys

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


# ============================================================
# Text cleaning / deterministic NLP helpers
# ============================================================

def clean_text(text: str) -> str:
    if not text:
        return ""

    if fix_text is not None:
        try:
            text = fix_text(text)
        except Exception:
            pass

    text = text.replace("\x00", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Common broken Word references/bookmarks.
    text = re.sub(
        r"Ошибка!\s*Закладка\s*не\s*опред[^\n]*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Error!\s*Bookmark\s*not\s*defined[^\n]*",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize line endings and whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


def detect_language_from_text(text: str) -> str:
    sample = clean_text(text[:12000]).lower()

    if len(sample) < 50:
        return "unknown"

    uz_lat_markers = [
        "o‘zbekiston",
        "o'zbekiston",
        "sog‘liq",
        "sog'liq",
        "bo‘yicha",
        "bo'yicha",
        "davolash",
        "kasallik",
        "shifokor",
        "tibbiy",
        "toshkent",
        "vazirligi",
        "tashxis",
        "profilaktika",
        "bemor",
    ]

    ru_markers = [
        "министерство",
        "здравоохранения",
        "республики",
        "клинический",
        "протокол",
        "стандарт",
        "лечение",
        "диагностика",
        "заболевание",
        "беременность",
        "пациент",
        "профилактика",
    ]

    uz_cyr_markers = [
        "соғлиқ",
        "сақлаш",
        "бўйича",
        "даволаш",
        "касаллик",
        "шифокор",
        "тиббий",
        "ташхис",
        "бемор",
        "вазирлиги",
    ]

    uz_cyr_chars = len(re.findall(r"[қўғҳ]", sample, flags=re.IGNORECASE))
    cyr_chars = len(re.findall(r"[а-яёқўғҳ]", sample, flags=re.IGNORECASE))
    latin_chars = len(re.findall(r"[a-z]", sample, flags=re.IGNORECASE))

    uz_lat_score = sum(1 for m in uz_lat_markers if m in sample)
    ru_score = sum(1 for m in ru_markers if m in sample)
    uz_cyr_score = sum(1 for m in uz_cyr_markers if m in sample)

    if uz_lat_score >= 2:
        return "uz_lat"

    if uz_cyr_chars >= 10 or uz_cyr_score >= 2:
        return "uz_cyr"

    if cyr_chars > latin_chars:
        if ru_score >= uz_cyr_score:
            return "ru"
        return "uz_cyr"

    if uz_lat_score >= 1:
        return "uz_lat"

    return "unknown"


def detect_language_from_path(path: str) -> str:
    p = path.lower().replace("\\", "/")
    filename = p.split("/")[-1]

    ru_markers = [
        " рус",
        "_рус",
        "-рус",
        "рус.",
        "русс",
        "русча",
        " russian",
        "_ru",
        "-ru",
        " ru.",
    ]

    uz_lat_markers = [
        " лат",
        "_лат",
        "-лат",
        "латин",
        "лотин",
        "лот",
        " latin",
        "_lat",
        "-lat",
        "uzb latin",
        "uzbek latin",
    ]

    uz_cyr_markers = [
        " кир",
        "_кир",
        "-кир",
        "кирил",
        "кирилл",
        " cyr",
        "_cyr",
        "-cyr",
        "uzb kir",
        "uzbek cyr",
        "узб кир",
    ]

    if any(m in filename for m in ru_markers):
        return "ru"

    if any(m in filename for m in uz_lat_markers):
        return "uz_lat"

    if any(m in filename for m in uz_cyr_markers):
        return "uz_cyr"

    if any(m in p for m in ru_markers):
        return "ru"

    if any(m in p for m in uz_lat_markers):
        return "uz_lat"

    if any(m in p for m in uz_cyr_markers):
        return "uz_cyr"

    if re.search(r"[қўғҳ]", filename, flags=re.IGNORECASE):
        return "uz_cyr"

    return "unknown"


def get_quality_flags(text: str, detected_language: str) -> List[str]:
    flags = []
    char_count = len(text or "")

    if char_count == 0:
        flags.append("empty_text")
    elif char_count < 500:
        flags.append("low_char_count")

    if detected_language == "unknown":
        flags.append("unknown_detected_language")

    # Too many replacement chars usually means broken decoding.
    if text and text.count("�") > 20:
        flags.append("encoding_replacement_chars")

    # Very low alphabet ratio can indicate garbage OCR.
    if text:
        letters = len(re.findall(r"[A-Za-zА-Яа-яЁёҚқЎўҒғҲҳ]", text))
        ratio = letters / max(len(text), 1)

        if ratio < 0.25 and char_count > 500:
            flags.append("low_letter_ratio")

    return flags


# ============================================================
# File/path classification
# ============================================================

STOP_FOLDERS = {
    "ворд",
    "word",
    "doc",
    "docx",
    "сайтга",
    "сайт",
    "pdf",
    "пдф",
    "нкп",
    "нкс",
    "мкп",
    "мкс",
    "протокол",
    "протоколы",
    "protocol",
    "protokol",
    "стандарт",
    "стандарты",
    "standard",
    "standart",
    "илова",
    "приложение",
}

REVIEW_MARKERS = [
    "рецензия",
    "тақриз",
    "такриз",
    "review",
    "эксперт",
    "expert",
    "хулоса",
    "заключение",
]


def stable_id(text: str, prefix: str = "id") -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def clean_part(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def is_temp_word_file(path: Path) -> bool:
    return path.name.startswith("~$")


def is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS and not is_temp_word_file(path)


def source_format_from_path(path: Path) -> str:
    ext = path.suffix.lower()

    if ext == ".docx":
        return "docx"

    if ext == ".doc":
        return "doc"

    if ext == ".pdf":
        return "pdf"

    return ext.replace(".", "")


def rel_parts_from_raw_path(path: Path) -> List[str]:
    p = Path(path)

    try:
        rel = p.relative_to(RAW_GUIDELINES_DIR)
    except Exception:
        try:
            rel = p.relative_to(PROJECT_ROOT / RAW_GUIDELINES_DIR)
        except Exception:
            rel = p

    return [x for x in rel.as_posix().split("/") if x]


def is_stop_folder(folder: str) -> bool:
    f = folder.lower().strip()
    f = re.sub(r"^[\d\s.+_-]+", "", f)
    return f in STOP_FOLDERS


def get_specialty_and_disease_path(path: Path) -> Tuple[str, str]:
    parts = rel_parts_from_raw_path(path)

    specialty = clean_part(parts[0]) if len(parts) >= 1 else "unknown_specialty"
    middle = parts[1:-1] if len(parts) >= 3 else []

    disease_parts = []

    for part in middle:
        if is_stop_folder(part):
            break

        disease_parts.append(clean_part(part))

    if not disease_parts:
        disease_parts = [Path(path).stem]

    disease_path = " / ".join(disease_parts)
    return specialty, disease_path


def is_review_path(path: Path) -> bool:
    p = str(path).lower().replace("\\", "/")
    return any(m in p for m in REVIEW_MARKERS)


def detect_doc_type(path: Path) -> str:
    p = str(path).lower().replace("\\", "/")
    filename = p.split("/")[-1]

    standard_markers = [
        "стандарт",
        "standart",
        "standard",
        "нкс",
        "мкс",
        "mks",
        "клиник стандарт",
    ]

    protocol_markers = [
        "нкп",
        "протокол",
        "protokol",
        "protocol",
        "мкп",
        "mpk",
        "mkp",
        "клиник протокол",
    ]

    if any(m in filename for m in standard_markers) or any(m in p for m in standard_markers):
        return "standard"

    if any(m in filename for m in protocol_markers) or any(m in p for m in protocol_markers):
        return "protocol"

    return "unknown"


def normalized_file_key(path: Path) -> str:
    name = path.stem.lower()

    remove_words = [
        "рус",
        "ру",
        "russian",
        "ru",
        "узб",
        "кирилл",
        "кирил",
        "кир",
        "cyr",
        "латин",
        "лотин",
        "лот",
        "latin",
        "lat",
        "нкп",
        "нкс",
        "мкп",
        "мкс",
        "протокол",
        "стандарт",
        "protocol",
        "standard",
        "protokol",
        "standart",
        "organized",
    ]

    for w in remove_words:
        name = name.replace(w, " ")

    name = re.sub(r"[\d_().,\-–—]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name or path.stem.lower()


def file_size_kb(path: Path) -> float:
    try:
        return round(path.stat().st_size / 1024, 2)
    except Exception:
        return 0.0


# ============================================================
# Command helpers
# ============================================================

def run_cmd(
    cmd: List[str],
    timeout: int = 600,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    started = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    result.elapsed_seconds = round(time.time() - started, 3)  # type: ignore[attr-defined]

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout: {result.stdout[-3000:]}\n"
            f"stderr: {result.stderr[-3000:]}"
        )

    return result


def check_command_exists(command: str) -> bool:
    return shutil.which(command) is not None or Path(command).exists()


def libreoffice_convert(
    source_path: Path,
    target_ext: str,
    out_dir: Path,
    timeout: int = 240,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        LIBREOFFICE_CMD,
        "--headless",
        "--convert-to",
        target_ext,
        "--outdir",
        str(out_dir),
        str(source_path),
    ]

    run_cmd(cmd, timeout=timeout)

    expected = out_dir / f"{source_path.stem}.{target_ext}"

    if expected.exists():
        return expected

    matches = list(out_dir.glob(f"{source_path.stem}*.{target_ext}"))

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"LibreOffice conversion output not found: {source_path} → {target_ext}"
    )


# ============================================================
# Regex helpers for validation later
# ============================================================

ICD_PATTERN = re.compile(r"\b[A-ZА-Я]\d{2}(?:\.\d{1,2})?\b", re.IGNORECASE)

MKB_MARKERS = [
    "мкб",
    "mkb",
    "icd",
    "мкх",
    "код",
    "code",
]


def find_icd_codes(text: str) -> List[str]:
    if not text:
        return []

    codes = []
    seen = set()

    for m in ICD_PATTERN.finditer(text):
        code = m.group(0).upper().replace(",", ".")

        if code not in seen:
            codes.append(code)
            seen.add(code)

    return codes


def has_mkb_marker(text: str) -> bool:
    s = (text or "").lower()
    return any(m in s for m in MKB_MARKERS)


# ============================================================
# Record builder
# ============================================================

def build_base_variant_record(path: Path) -> Dict[str, Any]:
    specialty, disease_path = get_specialty_and_disease_path(path)
    source_format = source_format_from_path(path)
    doc_type = detect_doc_type(path)
    language_hint = detect_language_from_path(str(path))

    disease_class_id = stable_id(f"{specialty}::{disease_path}", "dc")
    clinical_doc_group_id = stable_id(f"{specialty}::{disease_path}::{doc_type}", "cd")

    return {
        "variant_id": stable_id(str(path), "var"),
        "disease_class_id": disease_class_id,
        "clinical_doc_group_id": clinical_doc_group_id,
        "specialty": specialty,
        "disease_path": disease_path,
        "doc_type": doc_type,
        "language_hint": language_hint,
        "source_format": source_format,
        "source_extension": path.suffix.lower(),
        "source_path": path.as_posix(),
        "filename": path.name,
        "size_kb": file_size_kb(path),
        "normalized_file_key": normalized_file_key(path),
    }
