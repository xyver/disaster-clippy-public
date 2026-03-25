#!/usr/bin/env python3
"""
Preprocess scanned PDFs with OCRmyPDF.

Examples:
    python -m cli.ocr_preprocess --pdf "C:\\path\\document.pdf"
    python -m cli.ocr_preprocess --folder "C:\\path\\pdfs"
    python -m cli.ocr_preprocess --folder "C:\\path\\pdfs" --all
"""

import argparse
import json
import sys
from pathlib import Path

from offline_tools.ocr import (
    collect_pdf_text_stats,
    get_ocr_output_path,
    list_pdf_files,
    run_ocrmypdf,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR scanned PDFs with OCRmyPDF")
    parser.add_argument("--pdf", help="Single PDF to inspect/process")
    parser.add_argument("--folder", help="Folder containing PDFs")
    parser.add_argument("--all", action="store_true", help="OCR every discovered PDF, even if it already has text")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing .ocr.pdf output")
    parser.add_argument("--redo-ocr", action="store_true", help="Ask OCRmyPDF to redo OCR on an existing text layer")
    parser.add_argument("--language", default="eng", help="OCR language(s), e.g. eng or eng+fra")
    parser.add_argument("--min-chars", type=int, default=500, help="Skip OCR when extracted text already meets this threshold")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def discover_targets(args: argparse.Namespace) -> list[Path]:
    if args.pdf:
        pdf_path = Path(args.pdf)
        return [pdf_path] if pdf_path.exists() else []
    if args.folder:
        return list_pdf_files(args.folder, prefer_ocr=False)
    return []


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.pdf and not args.folder:
        parser.error("Provide either --pdf or --folder")

    targets = discover_targets(args)
    if not targets:
        print("No PDF files found", file=sys.stderr)
        return 1

    results = []
    for pdf_path in targets:
        stats = collect_pdf_text_stats(pdf_path)
        should_ocr = args.all or stats.total_chars < args.min_chars

        record = {
            "input_pdf": str(pdf_path),
            "output_pdf": str(get_ocr_output_path(pdf_path)),
            "stats": stats.to_dict(),
            "should_ocr": should_ocr,
        }

        if should_ocr:
            ocr_result = run_ocrmypdf(
                pdf_path,
                language=args.language,
                force=args.force,
                redo_ocr=args.redo_ocr,
            )
            record["ocr_result"] = ocr_result.to_dict()
        else:
            record["ocr_result"] = {
                "success": True,
                "input_pdf": str(pdf_path),
                "output_pdf": str(get_ocr_output_path(pdf_path)),
                "skipped": True,
                "skipped_reason": f"Existing text layer has {stats.total_chars} chars",
                "command": [],
                "error": "",
            }

        results.append(record)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for record in results:
            stats = record["stats"]
            result = record["ocr_result"]
            status = "OCRed" if result.get("success") and not result.get("skipped") else "Skipped"
            if not result.get("success"):
                status = "Failed"
            print(
                f"{status}: {Path(record['input_pdf']).name} | "
                f"pages={stats['page_count']} chars={stats['total_chars']} -> "
                f"{Path(record['output_pdf']).name}"
            )
            if result.get("skipped_reason"):
                print(f"  reason: {result['skipped_reason']}")
            if result.get("error"):
                print(f"  error: {result['error']}")

    failed = any(not record["ocr_result"].get("success") for record in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
