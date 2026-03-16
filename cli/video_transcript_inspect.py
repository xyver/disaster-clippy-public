"""
Video transcript inspector CLI.

Use this to validate transcript acquisition before wiring a source into
background jobs or indexing.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from admin.local_config import get_local_config
from offline_tools.transcript_acquisition import acquire_best_transcript
from offline_tools.video_analysis import prepare_video_transcripts
from offline_tools.youtube_transcript import fetch_youtube_transcript, parse_youtube_video_id
from offline_tools.video_models import VideoRecord


def print_report(source_path: Path, result, show_segments: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print(f"VIDEO TRANSCRIPT INSPECTION: {source_path.name}")
    print(f"{'=' * 60}\n")

    print(f"Source path: {source_path}")
    print(f"Video ID: {result.video_id}")
    print(f"Acquired: {'yes' if result.success else 'no'}")
    print(f"Requires ASR: {'yes' if result.requires_asr else 'no'}")

    if result.source_kind:
        print(f"Source kind: {result.source_kind}")
    if result.source_path:
        print(f"Scan root: {result.source_path}")
    if result.error:
        print(f"Error: {result.error}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    transcript = result.transcript
    if not transcript:
        print(f"\n{'=' * 60}\n")
        return

    print("\nTRANSCRIPT")
    print("-" * 40)
    print(f"Language: {transcript.language or '(unknown)'}")
    print(f"Original language: {transcript.original_language or transcript.language or '(unknown)'}")
    print(f"Segments: {len(transcript.segments)}")
    print(f"Full text length: {len(transcript.full_text)} chars")
    if transcript.metadata:
        source_file = transcript.metadata.get("source_file")
        if source_file:
            print(f"Source file: {source_file}")
        file_format = transcript.metadata.get("format")
        if file_format:
            print(f"Format: {file_format}")

    preview = transcript.full_text[:300].replace("\n", " ").strip()
    if preview:
        print(f"Preview: {preview}{'...' if len(transcript.full_text) > 300 else ''}")

    if show_segments and transcript.segments:
        print("\nSEGMENTS")
        print("-" * 40)
        for segment in transcript.segments[:20]:
            print(
                f"[{segment.segment_id}] "
                f"{segment.start_sec:.1f}-{segment.end_sec:.1f}s "
                f"{segment.text[:100]}"
            )
        if len(transcript.segments) > 20:
            print(f"... {len(transcript.segments) - 20} more segments")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect packaged transcript acquisition for a video source"
    )
    parser.add_argument("source_path", help="Path to a source folder")
    parser.add_argument("video_id", help="Video identifier to inspect")
    parser.add_argument("--title", default="", help="Optional human title")
    parser.add_argument("--source-url", default="", help="Optional live source URL")
    parser.add_argument("--language", default="", help="Optional known source language")
    parser.add_argument("--force-live", action="store_true", help="Fetch live YouTube transcript directly for debugging")
    parser.add_argument("--allow-live", action="store_true", help="Allow live YouTube transcript acquisition for this run without changing saved config")
    parser.add_argument("--prepare", action="store_true", help="Run the full preparation pipeline and write raw_data artifacts")
    parser.add_argument("--asr-video-path", default="", help="Optional local media path for ASR fallback")
    parser.add_argument("--chunk-duration", type=float, default=60.0, help="Chunk duration in seconds for preparation")
    parser.add_argument("--enrich-topics", action="store_true", help="Run topic enrichment and write topics_english.json")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("-s", "--show-segments", action="store_true", help="Print normalized segments")
    args = parser.parse_args()

    source_path = Path(args.source_path)
    if not source_path.exists():
        print(f"ERROR: source path not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    video = VideoRecord(
        video_id=args.video_id,
        title=args.title or args.video_id,
        source_url=args.source_url,
        language=args.language,
    )
    if args.force_live and args.source_url:
        live_video_id = parse_youtube_video_id(args.source_url) or args.video_id
        live_result = fetch_youtube_transcript(live_video_id, language=args.language or "en")
        if args.json:
            output = {
                "success": live_result.success,
                "video_id": live_result.video_id,
                "source_kind": live_result.transcript.source_kind if live_result.transcript else "",
                "requires_asr": not live_result.success,
                "error": live_result.error,
                "warnings": [],
                "source_path": str(source_path),
                "transcript": live_result.transcript.to_dict() if live_result.transcript else None,
                "from_cache": live_result.from_cache,
                "error_code": live_result.error_code,
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
            return
        class _ResultWrapper:
            video_id = live_result.video_id
            success = live_result.success
            source_kind = live_result.transcript.source_kind if live_result.transcript else ""
            requires_asr = not live_result.success
            error = live_result.error
            warnings = [f"from_cache={live_result.from_cache}", f"error_code={live_result.error_code}"]
            source_path = str(source_path)
            transcript = live_result.transcript
        print_report(source_path, _ResultWrapper(), show_segments=args.show_segments)
        if not live_result.success:
            sys.exit(2)
        return

    config = get_local_config()
    if args.prepare:
        result = prepare_video_transcripts(
            source_path,
            video,
            asr_video_path=args.asr_video_path or None,
            chunk_duration_seconds=args.chunk_duration,
            write_artifacts=True,
            allow_live_fetch=args.allow_live or None,
            enrich_topics=args.enrich_topics,
        )
        if args.json:
            output = {
                "success": result.get("success", False),
                "video_id": result.get("video_id"),
                "source_kind": result.get("source_kind", ""),
                "used_asr": result.get("used_asr", False),
                "requires_asr": result.get("requires_asr", False),
                "error": result.get("error", ""),
                "warnings": result.get("warnings", []),
                "artifacts": result.get("artifacts", {}),
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
            return
        print(f"\nPrepared video transcript artifacts for {result.get('video_id')}")
        for key, value in result.get("artifacts", {}).items():
            print(f"  {key}: {value}")
        if result.get("warnings"):
            print("Warnings:")
            for warning in result["warnings"]:
                print(f"  - {warning}")
        if not result.get("success", False):
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(2)
        return

    result = acquire_best_transcript(source_path, video, allow_live_fetch=args.allow_live or None)

    if args.json:
        output = {
            "success": result.success,
            "video_id": result.video_id,
            "source_kind": result.source_kind,
            "requires_asr": result.requires_asr,
            "error": result.error,
            "warnings": result.warnings,
            "source_path": result.source_path,
            "transcript": result.transcript.to_dict() if result.transcript else None,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    print_report(source_path, result, show_segments=args.show_segments)
    if args.source_url:
        effective_live = args.allow_live or config.is_live_transcript_fetch_enabled()
        print(f"Live fetch policy enabled: {effective_live}")
    if not result.success and result.requires_asr:
        sys.exit(2)


if __name__ == "__main__":
    main()
