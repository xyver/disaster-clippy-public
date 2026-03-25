"""
OCR helpers for scanned PDF preprocessing.

This keeps OCR as a separate, reviewable step:
- originals remain untouched
- OCR output is written as sibling ``.ocr.pdf`` files
- metadata/indexing can prefer OCR output automatically
"""

from __future__ import annotations

import shutil
import site
import subprocess
import sys
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union


OCR_SUFFIX = ".ocr.pdf"


@dataclass
class OCRTextStats:
    pdf_path: str
    page_count: int = 0
    total_chars: int = 0
    nonempty_pages: int = 0
    text_layer_detected: bool = False

    def to_dict(self) -> Dict[str, Union[str, int, bool]]:
        return asdict(self)


@dataclass
class OCRResult:
    success: bool
    input_pdf: str
    output_pdf: str
    skipped: bool = False
    skipped_reason: str = ""
    command: List[str] = None
    error: str = ""

    def __post_init__(self) -> None:
        if self.command is None:
            self.command = []

    def to_dict(self) -> Dict[str, Union[str, bool, List[str]]]:
        return asdict(self)


def get_ocrmypdf_executable() -> Optional[str]:
    """Return the installed ocrmypdf executable if available."""
    candidates = ["ocrmypdf", "ocrmypdf.exe"]
    script_dirs = []

    try:
        script_dirs.append(Path(sys.executable).resolve().parent / "Scripts")
    except Exception:
        pass

    try:
        versioned_user_scripts = (
            Path(site.getuserbase())
            / f"Python{sys.version_info.major}{sys.version_info.minor}"
            / "Scripts"
        )
        script_dirs.append(versioned_user_scripts)
    except Exception:
        pass

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        for script_dir in script_dirs:
            candidate_path = script_dir / candidate
            if candidate_path.exists():
                return str(candidate_path)
    return None


def _existing_paths(paths: List[Path]) -> List[str]:
    seen = []
    for path in paths:
        try:
            resolved = str(path)
        except Exception:
            continue
        if path.exists() and resolved not in seen:
            seen.append(resolved)
    return seen


def build_ocr_environment() -> Dict[str, str]:
    """Build an environment with likely OCR tool install paths on Windows."""
    env = os.environ.copy()
    extra_paths = _existing_paths([
        Path(sys.executable).resolve().parent / "Scripts",
        Path(site.getuserbase()) / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts",
        Path(r"C:\Program Files\Tesseract-OCR"),
        Path(r"C:\Program Files\gs\gs10.07.0\bin"),
        Path(r"C:\ProgramData\chocolatey\bin"),
    ])
    current_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(extra_paths + [current_path]) if current_path else os.pathsep.join(extra_paths)
    return env


def is_ocr_pdf(pdf_path: Union[str, Path]) -> bool:
    """Return True when the file looks like a generated sibling OCR PDF."""
    return Path(pdf_path).name.lower().endswith(OCR_SUFFIX)


def get_ocr_output_path(pdf_path: Union[str, Path]) -> Path:
    """Return the sibling OCR PDF path for an input PDF."""
    pdf_path = Path(pdf_path)
    if is_ocr_pdf(pdf_path):
        return pdf_path
    return pdf_path.with_name(f"{pdf_path.stem}{OCR_SUFFIX}")


def get_base_pdf_name(pdf_path: Union[str, Path]) -> str:
    """Normalize original and OCR derivative names to the same logical key."""
    pdf_path = Path(pdf_path)
    name = pdf_path.name
    if name.lower().endswith(OCR_SUFFIX):
        return f"{name[:-len(OCR_SUFFIX)]}.pdf"
    return name


def collect_pdf_text_stats(pdf_path: Union[str, Path]) -> OCRTextStats:
    """
    Inspect a PDF and estimate whether it already has a usable text layer.

    We intentionally keep this lightweight so it can be used as a fast gate
    before OCR and as a status signal in admin tools.
    """
    pdf_path = Path(pdf_path)
    stats = OCRTextStats(pdf_path=str(pdf_path))

    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        stats.page_count = doc.page_count
        for page in doc:
            text = page.get_text().strip()
            if text:
                stats.nonempty_pages += 1
                stats.total_chars += len(text)
        doc.close()
    except Exception:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            stats.page_count = len(reader.pages)
            for page in reader.pages:
                text = (page.extract_text() or "").strip()
                if text:
                    stats.nonempty_pages += 1
                    stats.total_chars += len(text)
        except Exception:
            return stats

    stats.text_layer_detected = stats.total_chars > 0 and stats.nonempty_pages > 0
    return stats


def pdf_has_usable_text(pdf_path: Union[str, Path], min_chars: int = 500) -> bool:
    """Return True when extracted text appears sufficient to skip OCR."""
    stats = collect_pdf_text_stats(pdf_path)
    return stats.total_chars >= min_chars


def list_pdf_files(
    source: Union[str, Path],
    *,
    prefer_ocr: bool = False,
    include_ocr_derivatives: bool = False,
) -> List[Path]:
    """
    List PDFs from a file or folder, optionally collapsing originals/OCR pairs.

    When ``prefer_ocr`` is True, ``foo.ocr.pdf`` is returned instead of
    ``foo.pdf`` when both exist.
    """
    source = Path(source)
    if source.is_file():
        return [source] if source.suffix.lower() == ".pdf" else []
    if not source.is_dir():
        return []

    pdf_files = sorted(source.glob("*.pdf"), key=lambda path: path.name.lower())
    if include_ocr_derivatives:
        return pdf_files

    grouped: Dict[str, Dict[str, Path]] = {}
    for pdf_path in pdf_files:
        base_name = get_base_pdf_name(pdf_path)
        group = grouped.setdefault(base_name, {})
        if is_ocr_pdf(pdf_path):
            group["ocr"] = pdf_path
        else:
            group["original"] = pdf_path

    selected: List[Path] = []
    for base_name in sorted(grouped.keys()):
        group = grouped[base_name]
        if prefer_ocr and group.get("ocr"):
            selected.append(group["ocr"])
        elif group.get("original"):
            selected.append(group["original"])
        elif group.get("ocr"):
            selected.append(group["ocr"])

    return selected


def run_ocrmypdf(
    input_pdf: Union[str, Path],
    output_pdf: Union[str, Path] = None,
    *,
    language: str = "eng",
    force: bool = False,
    redo_ocr: bool = False,
    deskew: bool = True,
    optimize: int = 1,
) -> OCRResult:
    """
    Run OCRmyPDF against a single file and create a searchable sibling PDF.
    """
    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf) if output_pdf else get_ocr_output_path(input_pdf)

    executable = get_ocrmypdf_executable()
    if not executable:
        return OCRResult(
            success=False,
            input_pdf=str(input_pdf),
            output_pdf=str(output_pdf),
            error="ocrmypdf executable not found on PATH",
        )

    if output_pdf.exists() and not force:
        return OCRResult(
            success=True,
            input_pdf=str(input_pdf),
            output_pdf=str(output_pdf),
            skipped=True,
            skipped_reason="OCR output already exists",
        )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    command = [
        executable,
        "--skip-text",
        "--optimize",
        str(optimize),
        "--language",
        language,
    ]
    if deskew:
        command.append("--deskew")
    if redo_ocr:
        command.append("--redo-ocr")
    if force and output_pdf.exists():
        command.append("--force-ocr")

    command.extend([str(input_pdf), str(output_pdf)])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=build_ocr_environment(),
        )
    except Exception as exc:
        return OCRResult(
            success=False,
            input_pdf=str(input_pdf),
            output_pdf=str(output_pdf),
            command=command,
            error=str(exc),
        )

    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        return OCRResult(
            success=False,
            input_pdf=str(input_pdf),
            output_pdf=str(output_pdf),
            command=command,
            error=stderr or f"ocrmypdf exited with code {completed.returncode}",
        )

    return OCRResult(
        success=True,
        input_pdf=str(input_pdf),
        output_pdf=str(output_pdf),
        command=command,
    )
