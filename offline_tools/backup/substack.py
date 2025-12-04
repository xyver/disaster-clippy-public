"""
Substack HTML Backup

Downloads Substack newsletter posts to local HTML files for offline browsing.
Uses the same CSV export + session cookie approach as the scraper.

Usage:
    python substack_backup.py posts.csv https://thebarracks.substack.com --limit 50
    python substack_backup.py posts.csv https://thebarracks.substack.com --all
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class SubstackBackup:
    """
    Backs up Substack newsletter posts to local HTML files.
    """

    def __init__(self, csv_path: str, newsletter_url: str, backup_path: str,
                 session_cookie: str = None, include_paid: bool = True):
        self.csv_path = csv_path
        self.newsletter_url = newsletter_url.rstrip("/")
        self.backup_path = Path(backup_path)
        self.session_cookie = session_cookie or os.getenv("SUBSTACK_SESSION_COOKIE")
        self.include_paid = include_paid

        # Extract source name from URL
        parsed = urlparse(newsletter_url)
        self.source_id = parsed.netloc.split(".")[0]

        # Setup directories
        self.backup_dir = self.backup_path / self.source_id
        self.pages_dir = self.backup_dir / "pages"
        self.assets_dir = self.backup_dir / "assets"

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)

        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        if self.session_cookie:
            self.session.cookies.set("substack.sid", self.session_cookie)

        # Load or create manifest
        self.manifest_path = self.backup_dir / "manifest.json"
        self.manifest = self._load_manifest()

        # Track stats
        self.pages_saved = 0
        self.pages_skipped = 0
        self.errors = []

    def _load_manifest(self) -> dict:
        """Load existing manifest or create new one"""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "source_id": self.source_id,
            "base_url": self.newsletter_url,
            "scraper_type": "substack",
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
            "pages": {},
            "total_pages": 0
        }

    def _save_manifest(self):
        """Save manifest to disk"""
        self.manifest["last_updated"] = datetime.now().isoformat()
        self.manifest["total_pages"] = len(self.manifest["pages"])
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def _load_posts_csv(self) -> list:
        """Load and parse the posts CSV file"""
        posts = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only include published posts
                if row.get('is_published') != 'true':
                    continue

                # Skip paid posts if no auth
                audience = row.get('audience', 'everyone')
                if audience == 'only_paid' and not self.session_cookie:
                    continue

                # Extract slug from post_id
                post_id = row.get('post_id', '')
                if '.' in post_id:
                    slug = post_id.split('.', 1)[1]
                else:
                    slug = post_id

                if not slug:
                    continue

                posts.append({
                    'slug': slug,
                    'title': row.get('title', ''),
                    'subtitle': row.get('subtitle', ''),
                    'post_date': row.get('post_date', ''),
                    'audience': audience,
                    'url': f"{self.newsletter_url}/p/{slug}"
                })

        return posts

    def _url_to_filename(self, slug: str) -> str:
        """Convert slug to safe filename"""
        safe_name = slug.replace('-', '_')
        if len(safe_name) > 80:
            safe_name = safe_name[:80] + "_" + hashlib.md5(slug.encode()).hexdigest()[:8]
        return safe_name + ".html"

    def _process_html(self, html: str, title: str) -> str:
        """Clean up HTML for offline viewing"""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove scripts, tracking, iframes
        for tag in soup.find_all(['script', 'iframe', 'noscript']):
            tag.decompose()

        # Remove subscription/paywall widgets
        selectors_to_remove = [
            '.subscribe-widget', '.subscription-widget', '.paywall',
            '.modal', '.popup', '[data-component="PaywallBar"]',
            '.footer-wrap', '.post-footer', '.comments-section'
        ]
        for selector in selectors_to_remove:
            for elem in soup.select(selector):
                elem.decompose()

        # Add offline notice at top
        body = soup.find('body')
        if body:
            notice = soup.new_tag('div')
            notice['style'] = 'background: #fff3cd; padding: 10px; text-align: center; font-size: 14px;'
            notice.string = f'Offline backup - {title}'
            body.insert(0, notice)

        return str(soup)

    def backup_post(self, post: dict) -> bool:
        """Backup a single post. Returns True if saved, False if skipped/error."""
        url = post['url']
        title = post['title'] or post['slug']

        # Skip if already backed up
        if url in self.manifest.get("pages", {}):
            return False

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Check for paywall
            if 'Keep reading with a 7-day free trial' in response.text:
                self.errors.append(f"Paywall: {title}")
                return False

            # Process HTML
            processed_html = self._process_html(response.text, title)

            # Save file
            filename = self._url_to_filename(post['slug'])
            filepath = self.pages_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(processed_html)

            # Update manifest
            self.manifest["pages"][url] = {
                "title": title,
                "subtitle": post.get('subtitle', ''),
                "filename": filename,
                "post_date": post.get('post_date', ''),
                "audience": post.get('audience', 'everyone'),
                "saved_at": datetime.now().isoformat(),
                "size": len(processed_html)
            }

            return True

        except Exception as e:
            self.errors.append(f"Error {title}: {str(e)}")
            return False

    def _create_index_html(self):
        """Create browsable index.html"""
        # Sort pages by date (newest first)
        sorted_pages = sorted(
            self.manifest["pages"].items(),
            key=lambda x: x[1].get("post_date", ""),
            reverse=True
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.source_id} - Offline Archive</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f7f7f7;
            color: #333;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #ff6719;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        .stats {{
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stats p {{ margin: 5px 0; }}
        .post-list {{ list-style: none; padding: 0; }}
        .post-item {{
            background: white;
            margin-bottom: 10px;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .post-item:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
        .post-title {{
            font-size: 1.1em;
            font-weight: 600;
            color: #ff6719;
            text-decoration: none;
            display: block;
            margin-bottom: 5px;
        }}
        .post-title:hover {{ text-decoration: underline; }}
        .post-subtitle {{ color: #666; font-size: 0.95em; margin-bottom: 5px; }}
        .post-meta {{ color: #999; font-size: 0.85em; }}
        .paid-badge {{
            background: #ff6719;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.75em;
            margin-left: 8px;
        }}
    </style>
</head>
<body>
    <h1>{self.source_id}</h1>
    <div class="stats">
        <p><strong>Source:</strong> <a href="{self.newsletter_url}">{self.newsletter_url}</a></p>
        <p><strong>Posts archived:</strong> {len(self.manifest['pages'])}</p>
        <p><strong>Last updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    <ul class="post-list">
"""

        for url, info in sorted_pages:
            date_str = info.get('post_date', '')[:10] if info.get('post_date') else ''
            subtitle = info.get('subtitle', '')
            is_paid = info.get('audience') == 'only_paid'
            paid_badge = '<span class="paid-badge">PAID</span>' if is_paid else ''

            html += f'''        <li class="post-item">
            <a class="post-title" href="pages/{info['filename']}">{info['title']}{paid_badge}</a>
            {f'<div class="post-subtitle">{subtitle}</div>' if subtitle else ''}
            <div class="post-meta">{date_str}</div>
        </li>
'''

        html += """    </ul>
</body>
</html>
"""
        with open(self.backup_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(html)

    def run(self, limit: int = None, progress_callback=None):
        """
        Run the backup process.

        Args:
            limit: Max posts to backup (None = all)
            progress_callback: Optional function(current, total, message)

        Returns:
            dict with results
        """
        posts = self._load_posts_csv()
        if limit:
            posts = posts[:limit]

        total = len(posts)
        print(f"Backing up {total} posts from {self.source_id}...")
        print(f"Output: {self.backup_dir}")
        if self.session_cookie:
            print("(Using session cookie for paid content)")
        print()

        for i, post in enumerate(posts):
            title = post['title'][:50] or post['slug']

            if progress_callback:
                progress_callback(i + 1, total, title)

            # Check if already backed up
            if post['url'] in self.manifest.get("pages", {}):
                self.pages_skipped += 1
                print(f"  [{i+1}/{total}] Skipped (exists): {title}")
                continue

            print(f"  [{i+1}/{total}] Backing up: {title}")

            if self.backup_post(post):
                self.pages_saved += 1
            else:
                self.pages_skipped += 1

            # Rate limit - be gentle with Substack
            time.sleep(1.5)

            # Save manifest periodically
            if (i + 1) % 20 == 0:
                self._save_manifest()

        # Final save
        self._save_manifest()
        self._create_index_html()

        print()
        print(f"Backup complete!")
        print(f"  New pages saved: {self.pages_saved}")
        print(f"  Skipped: {self.pages_skipped}")
        print(f"  Total in archive: {len(self.manifest['pages'])}")
        if self.errors:
            print(f"  Errors: {len(self.errors)}")
            for err in self.errors[:5]:
                print(f"    - {err}")

        return {
            "success": True,
            "pages_saved": self.pages_saved,
            "pages_skipped": self.pages_skipped,
            "total_archived": len(self.manifest["pages"]),
            "errors": self.errors,
            "output_dir": str(self.backup_dir)
        }


def main():
    parser = argparse.ArgumentParser(description="Backup Substack newsletter to HTML")
    parser.add_argument("csv_path", help="Path to posts.csv from Substack export")
    parser.add_argument("newsletter_url", help="Newsletter URL (e.g., https://example.substack.com)")
    parser.add_argument("--backup-path", default=os.getenv("BACKUP_PATH", "./backups"),
                        help="Base backup directory")
    parser.add_argument("--limit", type=int, help="Max posts to backup")
    parser.add_argument("--all", action="store_true", help="Backup all posts (ignore limit)")

    args = parser.parse_args()

    limit = None if args.all else (args.limit or 50)

    backup = SubstackBackup(
        csv_path=args.csv_path,
        newsletter_url=args.newsletter_url,
        backup_path=args.backup_path
    )

    result = backup.run(limit=limit)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
