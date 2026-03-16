"""
YouTube transcript retrieval helpers.

This is an optional online acquisition path used only when policy allows.
It is inspired by the fallback strategy seen in Sluice, but adapted for
Disaster Clippy's Python pipeline.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests

from .video_models import TranscriptDocument, TranscriptSegment


_TRANSCRIPT_CACHE: Dict[str, Tuple[float, "YouTubeTranscriptFetchResult"]] = {}
_CACHE_TTL_SECONDS = 5 * 60

_RE_XML_STANDARD = re.compile(r'<text start="([^"]*)" dur="([^"]*)">([^<]*)</text>')
_RE_XML_ASR = re.compile(r'<p t="(\d+)" d="(\d+)"[^>]*>([\s\S]*?)</p>')
_RE_XML_ASR_SEGMENT = re.compile(r"<s[^>]*>([^<]*)</s>")

_INNERTUBE_CLIENTS = (
    {
        "name": "ANDROID",
        "headers": {
            "Content-Type": "application/json",
            "User-Agent": "com.google.android.youtube/19.09.37 (Linux; Android 13)",
        },
        "context": {
            "client": {
                "clientName": "ANDROID",
                "clientVersion": "19.09.37",
                "androidSdkVersion": 33,
            }
        },
    },
    {
        "name": "WEB",
        "headers": {
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Cookie": "CONSENT=PENDING+987; SOCS=CAESEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg",
        },
        "context": {
            "client": {
                "clientName": "WEB",
                "clientVersion": "2.20240101.00.00",
            }
        },
    },
)


@dataclass
class YouTubeTranscriptFetchResult:
    success: bool
    video_id: str
    transcript: Optional[TranscriptDocument] = None
    error: str = ""
    error_code: str = ""
    language: str = ""
    from_cache: bool = False


def parse_youtube_video_id(url: str) -> Optional[str]:
    """Extract a YouTube video ID from common URL shapes."""
    if not url or not isinstance(url, str):
        return None

    normalized = url.strip()
    if not re.match(r"^https?://", normalized, re.IGNORECASE):
        normalized = f"https://{normalized}"

    try:
        parsed = urlparse(normalized)
    except Exception:
        return None

    hostname = parsed.hostname.lower() if parsed.hostname else ""
    valid_hostnames = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
    if hostname not in valid_hostnames:
        return None

    video_id = None
    if hostname == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/", 1)[0]
    elif "/watch" in parsed.path:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    elif "/embed/" in parsed.path:
        video_id = parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
    elif "/v/" in parsed.path:
        video_id = parsed.path.split("/v/", 1)[1].split("/", 1)[0]

    if not video_id or not re.fullmatch(r"[a-zA-Z0-9_-]{11}", video_id):
        return None
    return video_id


def fetch_youtube_transcript(video_id: str, language: str = "en") -> YouTubeTranscriptFetchResult:
    """Fetch a YouTube transcript using InnerTube fallbacks."""
    cached = _TRANSCRIPT_CACHE.get(video_id)
    now = time.time()
    if cached and cached[0] > now:
        result = cached[1]
        return YouTubeTranscriptFetchResult(
            success=result.success,
            video_id=result.video_id,
            transcript=result.transcript,
            error=result.error,
            error_code=result.error_code,
            language=result.language,
            from_cache=True,
        )

    try:
        items = _fetch_transcript_innertube(video_id, language=language)
        if not items:
            result = YouTubeTranscriptFetchResult(
                success=False,
                video_id=video_id,
                error="No transcript available for this video",
                error_code="no_transcript",
                language=language,
            )
            _TRANSCRIPT_CACHE[video_id] = (now + _CACHE_TTL_SECONDS, result)
            return result

        segments: List[TranscriptSegment] = []
        for index, item in enumerate(items):
            text = str(item["text"]).strip()
            if not text:
                continue
            start = float(item["offset"])
            end = start + float(item["duration"])
            segments.append(
                TranscriptSegment(
                    segment_id=f"{video_id}_{index + 1}",
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                    text=text,
                    language=language,
                    source_kind="online_caption",
                )
            )

        transcript = TranscriptDocument(
            video_id=video_id,
            language=language,
            source_kind="online_caption",
            full_text="\n\n".join(segment.text for segment in segments if segment.text.strip()),
            segments=segments,
            original_language=language,
            retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={"provider": "youtube_innertube"},
        )
        result = YouTubeTranscriptFetchResult(
            success=True,
            video_id=video_id,
            transcript=transcript,
            language=language,
        )
        _TRANSCRIPT_CACHE[video_id] = (now + _CACHE_TTL_SECONDS, result)
        return result
    except Exception as error:
        message = str(error)
        error_code = _categorize_youtube_error(message)
        result = YouTubeTranscriptFetchResult(
            success=False,
            video_id=video_id,
            error=message,
            error_code=error_code,
            language=language,
        )
        _TRANSCRIPT_CACHE[video_id] = (now + _CACHE_TTL_SECONDS, result)
        return result


def _fetch_transcript_innertube(video_id: str, language: str = "en") -> List[Dict[str, float | str]]:
    last_error: Optional[Exception] = None
    for client in _INNERTUBE_CLIENTS:
        try:
            result = _try_innertube_client(video_id, language, client)
            if result:
                return result
        except Exception as error:
            last_error = error
    if last_error:
        raise last_error
    raise RuntimeError("All transcript fetch strategies failed")


def _try_innertube_client(
    video_id: str,
    language: str,
    client: Dict[str, object],
) -> List[Dict[str, float | str]]:
    response = requests.post(
        "https://www.youtube.com/youtubei/v1/player",
        headers=client["headers"],  # type: ignore[arg-type]
        json={
            "context": {
                "client": {
                    **client["context"]["client"],  # type: ignore[index]
                    "hl": language,
                    "gl": "US",
                }
            },
            "videoId": video_id,
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    captions = (
        data.get("captions", {})
        .get("playerCaptionsTracklistRenderer", {})
    )
    tracks = captions.get("captionTracks") or []
    if not tracks:
        playability = data.get("playabilityStatus", {}).get("status", "unknown")
        raise RuntimeError(f"Transcript is disabled on this video ({client['name']}: {playability})")

    track = (
        next((t for t in tracks if t.get("languageCode") == language), None)
        or next((t for t in tracks if str(t.get("languageCode", "")).startswith(f"{language}-")), None)
        or next((t for t in tracks if t.get("kind") == "asr"), None)
        or tracks[0]
    )
    base_url = track.get("baseUrl")
    if not base_url:
        raise RuntimeError("Caption track missing baseUrl")

    transcript_response = requests.get(
        base_url,
        headers={
            **client["headers"],  # type: ignore[arg-type]
            "Accept-Language": language,
        },
        timeout=15,
    )
    transcript_response.raise_for_status()
    xml = transcript_response.text
    if not xml:
        raise RuntimeError("Empty transcript response")

    standard_results = list(_RE_XML_STANDARD.finditer(xml))
    if standard_results:
        return [
            {
                "text": html.unescape(match.group(3) or "").strip(),
                "duration": float(match.group(2) or 0.0),
                "offset": float(match.group(1) or 0.0),
            }
            for match in standard_results
            if (match.group(3) or "").strip()
        ]

    asr_results = list(_RE_XML_ASR.finditer(xml))
    if asr_results:
        output: List[Dict[str, float | str]] = []
        for match in asr_results:
            raw_text = match.group(3) or ""
            segment_matches = list(_RE_XML_ASR_SEGMENT.finditer(raw_text))
            if segment_matches:
                text = "".join(m.group(1) or "" for m in segment_matches).strip()
            else:
                text = re.sub(r"<[^>]*>", "", raw_text).strip()
            text = html.unescape(text)
            if not text:
                continue
            output.append(
                {
                    "text": text,
                    "duration": float(match.group(2) or 0.0) / 1000.0,
                    "offset": float(match.group(1) or 0.0) / 1000.0,
                }
            )
        return output

    raise RuntimeError("No transcript content found in response")


def _categorize_youtube_error(message: str) -> str:
    message_lower = message.lower()
    if "disabled" in message_lower:
        return "disabled"
    if "private" in message_lower or "unavailable" in message_lower:
        return "unavailable"
    if "empty transcript" in message_lower or "no transcript" in message_lower:
        return "no_transcript"
    if "timed out" in message_lower or "timeout" in message_lower:
        return "timeout"
    return "fetch_failed"
