"""
ZIM File Utilities
Reusable functions for ZIM file inspection, indexing, and content extraction
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ZIMInspectionResult:
    """Results from ZIM file inspection"""
    file_path: str
    file_size_mb: float
    article_count: int

    # Header metadata
    title: Optional[str] = None
    description: Optional[str] = None
    creator: Optional[str] = None
    publisher: Optional[str] = None
    language: Optional[str] = None
    license: Optional[str] = None
    source_url: Optional[str] = None
    date: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Content analysis
    mimetype_distribution: Dict[str, int] = field(default_factory=dict)
    namespace_distribution: Dict[str, int] = field(default_factory=dict)

    # Text analysis
    html_article_count: int = 0
    indexable_count: int = 0  # Articles with >= min_text_length chars
    skipped_count: int = 0    # Articles with < min_text_length chars
    avg_text_length: float = 0
    min_text_length_used: int = 50

    # Content type detection
    has_video_content: bool = False
    has_pdf_content: bool = False
    video_article_count: int = 0

    # Sample articles
    sample_indexable: List[Dict] = field(default_factory=list)
    sample_skipped: List[Dict] = field(default_factory=list)
    sample_videos: List[Dict] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    content_type: str = "website"  # website, video, pdf, mixed

    # Error info
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "file_path": self.file_path,
            "file_size_mb": self.file_size_mb,
            "article_count": self.article_count,
            "metadata": {
                "title": self.title,
                "description": self.description,
                "creator": self.creator,
                "publisher": self.publisher,
                "language": self.language,
                "license": self.license,
                "source_url": self.source_url,
                "date": self.date,
                "tags": self.tags,
            },
            "content_analysis": {
                "mimetype_distribution": self.mimetype_distribution,
                "namespace_distribution": self.namespace_distribution,
                "html_article_count": self.html_article_count,
                "indexable_count": self.indexable_count,
                "skipped_count": self.skipped_count,
                "avg_text_length": round(self.avg_text_length, 1),
                "skip_rate_percent": round(
                    (self.skipped_count / self.html_article_count * 100)
                    if self.html_article_count > 0 else 0, 1
                ),
            },
            "content_type": self.content_type,
            "has_video_content": self.has_video_content,
            "has_pdf_content": self.has_pdf_content,
            "video_article_count": self.video_article_count,
            "samples": {
                "indexable": self.sample_indexable[:5],
                "skipped": self.sample_skipped[:5],
                "videos": self.sample_videos[:5],
            },
            "recommendations": self.recommendations,
            "error": self.error,
        }


def inspect_zim_file(
    zim_path: str,
    scan_limit: int = 5000,
    min_text_length: int = 50,
    sample_size: int = 10,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> ZIMInspectionResult:
    """
    Inspect a ZIM file and return detailed analysis.

    Args:
        zim_path: Path to the ZIM file
        scan_limit: Maximum articles to scan (for large ZIMs)
        min_text_length: Minimum text length to consider indexable
        sample_size: Number of sample articles to include
        progress_callback: Optional callback(current, total, message)

    Returns:
        ZIMInspectionResult with full analysis
    """
    zim_path = Path(zim_path)

    # Initialize result
    result = ZIMInspectionResult(
        file_path=str(zim_path),
        file_size_mb=0,
        article_count=0,
        min_text_length_used=min_text_length
    )

    # Check file exists
    if not zim_path.exists():
        result.error = f"File not found: {zim_path}"
        return result

    result.file_size_mb = round(zim_path.stat().st_size / (1024 * 1024), 2)

    # Import zimply-core
    try:
        from zimply_core.zim_core import ZIMFile
    except ImportError:
        result.error = "zimply-core not installed. Run: pip install zimply-core"
        return result

    # Import text extractor
    try:
        from offline_tools.indexer import extract_text_from_html
    except ImportError:
        def extract_text_from_html(html):
            """Fallback text extractor"""
            import re
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

    try:
        zim = ZIMFile(str(zim_path), 'utf-8')
    except Exception as e:
        result.error = f"Failed to open ZIM file: {e}"
        return result

    # Extract header metadata
    header = zim.header_fields
    result.article_count = header.get('articleCount', 0)
    result.title = header.get('Title') or header.get('Name')
    result.description = header.get('Description')
    result.creator = header.get('Creator')
    result.publisher = header.get('Publisher')
    result.language = header.get('Language')
    result.license = header.get('License')
    result.source_url = header.get('Source')
    result.date = header.get('Date')

    # Parse tags
    tags_str = header.get('Tags', '')
    if tags_str:
        result.tags = [t.strip() for t in tags_str.split(';') if t.strip()]

    # Analyze content
    mimetype_counts = Counter()
    namespace_counts = Counter()
    text_lengths = []
    html_articles = []
    video_articles = []

    actual_scan = min(result.article_count, scan_limit)

    if progress_callback:
        progress_callback(0, actual_scan, "Scanning ZIM contents...")

    for i in range(actual_scan):
        if progress_callback and i % 100 == 0:
            progress_callback(i, actual_scan, f"Analyzed {i}/{actual_scan} articles...")

        try:
            article = zim.get_article_by_id(i)
            if article is None:
                continue

            url = getattr(article, 'url', '') or ''
            title = getattr(article, 'title', '') or ''
            mimetype = str(getattr(article, 'mimetype', 'unknown'))

            # Count mimetype
            mimetype_counts[mimetype] += 1

            # Count namespace
            namespace = url.split('/')[0] if '/' in url else (url[0] if url else 'unknown')
            namespace_counts[namespace] += 1

            # Analyze HTML content
            if 'text/html' in mimetype:
                content = article.data
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='ignore')

                text = extract_text_from_html(content)
                text_len = len(text)
                text_lengths.append(text_len)

                # Check for video content
                is_video = any(x in url.lower() or x in content.lower()
                              for x in ['video', 'youtube', 'watch', 'player', '.mp4', '.webm', 'vimeo'])

                article_info = {
                    'url': url[:100],
                    'title': title[:100],
                    'text_length': text_len,
                    'preview': text[:150] if text else ''
                }

                if is_video:
                    video_articles.append(article_info)
                    result.has_video_content = True
                else:
                    html_articles.append(article_info)

            # Check for PDF content
            elif 'application/pdf' in mimetype:
                result.has_pdf_content = True

        except Exception:
            continue

    # Close ZIM file
    try:
        zim.close()
    except Exception:
        pass

    # Update result with analysis
    result.mimetype_distribution = dict(mimetype_counts.most_common(15))
    result.namespace_distribution = dict(namespace_counts.most_common(10))
    result.html_article_count = len(text_lengths)
    result.video_article_count = len(video_articles)

    if text_lengths:
        result.indexable_count = sum(1 for l in text_lengths if l >= min_text_length)
        result.skipped_count = sum(1 for l in text_lengths if l < min_text_length)
        result.avg_text_length = sum(text_lengths) / len(text_lengths)

    # Sort samples by text length
    html_articles.sort(key=lambda x: x['text_length'], reverse=True)
    video_articles.sort(key=lambda x: x['text_length'], reverse=True)

    # Get samples
    result.sample_indexable = [a for a in html_articles if a['text_length'] >= min_text_length][:sample_size]
    result.sample_skipped = [a for a in html_articles if a['text_length'] < min_text_length][:sample_size]
    result.sample_videos = video_articles[:sample_size]

    # Determine content type
    if result.video_article_count > result.html_article_count * 0.5:
        result.content_type = "video"
    elif result.has_pdf_content and 'application/pdf' in result.mimetype_distribution:
        pdf_count = result.mimetype_distribution.get('application/pdf', 0)
        if pdf_count > result.html_article_count * 0.3:
            result.content_type = "pdf"
    elif result.has_video_content and result.has_pdf_content:
        result.content_type = "mixed"
    else:
        result.content_type = "website"

    # Generate recommendations
    _generate_recommendations(result)

    if progress_callback:
        progress_callback(actual_scan, actual_scan, "Analysis complete")

    return result


def _generate_recommendations(result: ZIMInspectionResult) -> None:
    """Generate recommendations based on inspection results"""

    if result.html_article_count == 0:
        result.recommendations.append(
            "WARNING: No HTML articles found. This ZIM may contain only binary content."
        )
        return

    skip_rate = result.skipped_count / result.html_article_count * 100 if result.html_article_count > 0 else 0

    # High skip rate
    if skip_rate > 70:
        result.recommendations.append(
            f"HIGH SKIP RATE: {skip_rate:.0f}% of articles have <{result.min_text_length_used} chars. "
            "Consider lowering the minimum text threshold for this source."
        )

    if skip_rate > 50:
        result.recommendations.append(
            "Many articles have minimal text content. This is common for:"
        )
        if result.content_type == "video":
            result.recommendations.append(
                "  - Video archives (descriptions may be short)"
            )
            result.recommendations.append(
                "  - Consider extracting video titles + descriptions as searchable content"
            )
            result.recommendations.append(
                "  - Look for subtitle files (.srt, .vtt) that contain full transcripts"
            )
        elif result.content_type == "pdf":
            result.recommendations.append(
                "  - PDF collections (index pages have limited text)"
            )
            result.recommendations.append(
                "  - The HTML pages may just be navigation/index pages"
            )
        else:
            result.recommendations.append(
                "  - Navigation/index pages"
            )
            result.recommendations.append(
                "  - Image galleries with captions"
            )

    # Good indexability
    if skip_rate < 30 and result.indexable_count > 100:
        result.recommendations.append(
            f"GOOD: This ZIM has {result.indexable_count} indexable articles. "
            "Standard indexing should work well."
        )

    # Video content detected
    if result.has_video_content and result.video_article_count > 10:
        result.recommendations.append(
            f"VIDEO CONTENT: Found {result.video_article_count} video-related pages. "
            "Consider using video-aware indexing to capture descriptions and metadata."
        )

    # License check
    if not result.license:
        result.recommendations.append(
            "LICENSE: No license detected in ZIM metadata. "
            "Please verify licensing before distributing."
        )

    # Size warning for very large ZIMs
    if result.article_count > 100000:
        result.recommendations.append(
            f"LARGE ZIM: {result.article_count} articles. "
            "Indexing may take a while. Consider setting a limit."
        )


def get_zim_metadata(zim_path: str) -> Dict:
    """
    Quick extraction of just the ZIM metadata (no content analysis).
    Useful for populating source configuration.
    """
    try:
        from zimply_core.zim_core import ZIMFile
    except ImportError:
        return {"error": "zimply-core not installed"}

    zim_path = Path(zim_path)
    if not zim_path.exists():
        return {"error": f"File not found: {zim_path}"}

    try:
        zim = ZIMFile(str(zim_path), 'utf-8')
        header = zim.header_fields

        metadata = {
            "title": header.get('Title') or header.get('Name') or zim_path.stem,
            "description": header.get('Description') or "",
            "creator": header.get('Creator') or "",
            "publisher": header.get('Publisher') or "",
            "language": header.get('Language') or "",
            "license": header.get('License') or "",
            "source_url": header.get('Source') or "",
            "date": header.get('Date') or "",
            "article_count": header.get('articleCount', 0),
            "tags": [],
        }

        tags_str = header.get('Tags', '')
        if tags_str:
            metadata["tags"] = [t.strip() for t in tags_str.split(';') if t.strip()]

        zim.close()
        return metadata

    except Exception as e:
        return {"error": f"Failed to read ZIM: {e}"}


def find_zim_files(folder_path: str) -> List[Dict]:
    """
    Find all ZIM files in a folder (including subfolders).

    Returns list of dicts with file info.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"[find_zim_files] Folder does not exist: {folder_path}")
        return []

    zim_files = []
    seen_paths = set()  # Avoid duplicates from case variations

    def add_zim_file(zim_file: Path, in_subfolder: bool, source_id: str = None):
        """Add a ZIM file to the list if not already present"""
        path_str = str(zim_file)
        path_lower = path_str.lower()
        if path_lower in seen_paths:
            return
        seen_paths.add(path_lower)

        try:
            zim_files.append({
                "path": path_str,
                "name": zim_file.name,
                "size_mb": round(zim_file.stat().st_size / (1024 * 1024), 2),
                "source_id": source_id or zim_file.stem,
                "in_subfolder": in_subfolder
            })
        except Exception as e:
            print(f"[find_zim_files] Error reading {zim_file}: {e}")

    # Check root level - use iterdir for more explicit handling
    try:
        for item in folder.iterdir():
            if item.is_file() and item.suffix.lower() == '.zim':
                add_zim_file(item, in_subfolder=False)
            elif item.is_dir():
                # Check subfolders
                try:
                    for subitem in item.iterdir():
                        if subitem.is_file() and subitem.suffix.lower() == '.zim':
                            add_zim_file(subitem, in_subfolder=True, source_id=item.name)
                except PermissionError:
                    pass
    except PermissionError as e:
        print(f"[find_zim_files] Permission error: {e}")
    except Exception as e:
        print(f"[find_zim_files] Error scanning folder: {e}")

    print(f"[find_zim_files] Found {len(zim_files)} ZIM files in {folder_path}")
    return sorted(zim_files, key=lambda x: x["name"])
