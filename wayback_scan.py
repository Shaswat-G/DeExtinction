"""
Wayback brand evolution scanner for Colossal Biosciences (2021–2025).

Classes
-------
- WaybackCollector: query & download Wayback snapshots, clean HTML -> text, save dataset
- DiffAnalyzer: load texts, compute diffs/similarities, keyword trends, and change events

Usage
-----
python wayback_brand_scan.py
(then see ./data/ for saved texts and ./reports/ for CSV/markdown outputs)
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import pathlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import tz
from readability import Document
from waybackpy import WaybackMachineCDXServerAPI

# Handle optional rapidfuzz import
try:
    from rapidfuzz.distance import Cosine

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    import warnings

    warnings.warn("rapidfuzz not available, falling back to simpler similarity metric")

# ------------------------ Config ------------------------

BASE_DOMAIN = "https://colossal.com"
OUTPUT_DIR = pathlib.Path("data")
REPORT_DIR = pathlib.Path("reports")
USER_AGENT = "WaybackBrandScan/1.0 (research; contact: you@example.com)"
START = 20210101  # inclusive (YYYYMMDD)
END = 20251231  # inclusive
REQUEST_SLEEP = 1.0  # polite delay between downloads (s)
TIMEZONE = "Europe/Zurich"

# Pages to track (add/remove as needed)
CANONICAL_PATHS = [
    "/",  # home
    "/about/",  # about
    "/species/woolly-mammoth/",
    "/species/thylacine/",
    "/species/dodo/",
    "/species/dire-wolf/",
    "/blog/",
    "/news/",
    "/press/",
    "/foundation/",
    "/how-it-works/",
    "/careers/",
]

# Quarterly sampling: pick the first snapshot in each quarter per path
SAMPLING = "quarterly"  # "all" | "yearly" | "quarterly"

# Keyword buckets for narrative coding
KEYWORDS = {
    "de_extinction": ["de-extinction", "de extinction", "deextinction"],
    "functional_de_extinction": ["functional de-extinction", "proxy", "proxy species"],
    "climate_benefit": ["climate", "carbon", "permafrost", "ecosystem", "biodiversity"],
    "iucn_alignment": ["IUCN", "International Union for Conservation of Nature"],
    "animal_welfare": [
        "welfare",
        "ethics",
        "ethical",
        "suffering",
        "well-being",
        "wellbeing",
    ],
    "indigenous": ["indigenous", "iwi", "māori", "maori", "first nations", "tribal"],
    "hype_vs_caution": ["moonshot", "sci-fi", "hype", "risk", "caution", "concern"],
}

# Change detection threshold (cosine distance on token shingles)
SIGNIFICANT_CHANGE = 0.20  # 0 = identical, 1 = completely different

# Image analysis settings
HERO_IMAGE_SELECTORS = [
    "img.hero-image",
    ".hero img",
    ".banner img",
    "img[src*='hero']",
    "img[src*='banner']",
    ".main-image img",
    "header img:first-of-type",
]
IMAGE_HASH_THRESHOLD = 10  # Hamming distance threshold for "significant" image changes

# Tagline extraction patterns
TAGLINE_SELECTORS = [
    "h1",
    ".hero h1",
    ".hero h2",
    ".banner h1",
    ".tagline",
    ".slogan",
    "meta[name='description']",
    "meta[property='og:description']",
]

# ------------------------ Utilities ------------------------


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def ts_to_local(ts: datetime, zone: str = TIMEZONE) -> datetime:
    """Convert aware UTC datetime to local zone."""
    return ts.astimezone(tz.gettz(zone))


def quarter_key(yyyymmdd: str) -> str:
    y = int(yyyymmdd[:4])
    m = int(yyyymmdd[4:6])
    q = (m - 1) // 3 + 1
    return f"{y}-Q{q}"


def safe_slug(path: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-]+", "-", path.strip("/")).strip("-") or "home"


def html_to_text(url: str, html: str) -> str:
    """
    Clean HTML to readable article-like text.
    Combines readability-lxml and BeautifulSoup fallback to strip boilerplate.
    """
    try:
        doc = Document(html)
        main_html = doc.summary()
        soup = BeautifulSoup(main_html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "lxml")

    # Remove nav/footer/scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    for selector in ["nav", "footer", "header", "aside", "form", "svg", "img"]:
        for n in soup.select(selector):
            n.decompose()

    text = soup.get_text("\n", strip=True)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Attach URL at top for provenance
    return f"[SOURCE] {url}\n\n{text}".strip()


def cosine_distance(a: str, b: str) -> float:
    """
    Cosine distance using 3-gram character shingles for stability to small edits.
    Returns 0..1 (0 identical, 1 very different).
    """

    def shingles(s: str, n: int = 3) -> List[str]:
        s = re.sub(r"\s+", " ", s.lower())
        return [s[i : i + n] for i in range(max(len(s) - n + 1, 0))]

    A, B = shingles(a), shingles(b)
    if not A and not B:
        return 0.0

    if RAPIDFUZZ_AVAILABLE:
        return 1.0 - Cosine.normalized_similarity(A, B)
    else:
        # Fallback to simple Jaccard similarity
        set_a, set_b = set(A), set(B)
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard = intersection / union if union > 0 else 0.0
        return 1.0 - jaccard


def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def extract_hero_images(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract hero/banner image URLs from HTML soup."""
    images = []
    for selector in HERO_IMAGE_SELECTORS:
        try:
            for img in soup.select(selector):
                src = img.get("src")
                if src and isinstance(src, str):
                    # Convert relative URLs to absolute
                    if src.startswith("//"):
                        src = "https:" + src
                    elif src.startswith("/"):
                        src = base_url.rstrip("/") + src
                    elif not src.startswith("http"):
                        continue  # Skip data URLs, etc.
                    images.append(src)
        except Exception:
            continue
    return list(dict.fromkeys(images))  # Remove duplicates while preserving order


def get_image_hash(image_url: str, session: requests.Session) -> str:
    """Download image and compute perceptual hash."""
    try:
        # Handle optional pillow/imagehash imports
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            logging.warning("PIL/imagehash not available, skipping image analysis")
            return ""

        response = session.get(image_url, timeout=30, stream=True)
        response.raise_for_status()

        # Load image and compute hash
        image = Image.open(BytesIO(response.content))
        # Use difference hash (dHash) which is good for detecting structural changes
        dhash = imagehash.dhash(image)
        return str(dhash)
    except Exception as e:
        logging.warning(f"Failed to hash image {image_url}: {e}")
        return ""


def extract_taglines(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract taglines/slogans from HTML using various selectors."""
    taglines = {}

    for selector in TAGLINE_SELECTORS:
        try:
            if selector.startswith("meta"):
                # Handle meta tags
                if "name=" in selector:
                    attr_name = selector.split("name='")[1].split("'")[0]
                    meta = soup.find("meta", attrs={"name": attr_name})
                elif "property=" in selector:
                    attr_name = selector.split("property='")[1].split("'")[0]
                    meta = soup.find("meta", attrs={"property": attr_name})
                else:
                    continue

                if meta and hasattr(meta, "get"):
                    content = meta.get("content", "")
                    if content and isinstance(content, str):
                        content = content.strip()
                        if content:
                            taglines[f"meta_{attr_name}"] = content
            else:
                # Handle regular CSS selectors
                elements = soup.select(selector)
                for i, elem in enumerate(elements):
                    text = elem.get_text(strip=True)
                    if text and len(text) < 500:  # Reasonable tagline length
                        key = f"{selector}_{i}" if i > 0 else selector
                        taglines[key] = text
        except Exception:
            continue

    return taglines


# ------------------------ WaybackCollector ------------------------


@dataclass
class WaybackCollector:
    base_domain: str = BASE_DOMAIN
    start_ts: str = str(START)
    end_ts: str = str(END)
    user_agent: str = USER_AGENT
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self):
        self.session.headers.update({"User-Agent": self.user_agent})

    def list_snapshots(self, path: str) -> List[Tuple[str, str]]:
        """
        Return list of (timestamp, archive_url) for a given path between start/end.
        """
        try:
            url = self.base_domain.rstrip("/") + path
            cdx = WaybackMachineCDXServerAPI(
                url,
                start_timestamp=self.start_ts,
                end_timestamp=self.end_ts,
                user_agent=self.user_agent,
            )
            snaps = []
            for snap in cdx.snapshots():  # yields WaybackMachineCDXSnapshot
                # Keep only 200s (CDX already filtered usually, but safeguard)
                if getattr(snap, "statuscode", "200") == "200":
                    snaps.append((snap.timestamp, snap.archive_url))
            return snaps
        except Exception as e:
            logging.error(f"Failed to list snapshots for {path}: {e}")
            return []

    def sample_snapshots(
        self, snaps: List[Tuple[str, str]], mode: str = SAMPLING
    ) -> List[Tuple[str, str]]:
        if mode == "all":
            return snaps
        if mode == "yearly":
            # earliest per year
            chosen = {}
            for ts, url in snaps:
                y = ts[:4]
                if y not in chosen:
                    chosen[y] = (ts, url)
            return [chosen[y] for y in sorted(chosen)]
        if mode == "quarterly":
            chosen = {}
            for ts, url in snaps:
                qk = quarter_key(ts)
                if qk not in chosen:
                    chosen[qk] = (ts, url)

            # Return in chronological order
            def sort_key(x: Tuple[str, str]) -> Tuple[int, int]:
                return (int(x[0][:4]), int((int(x[0][4:6]) - 1) // 3 + 1))

            return sorted(chosen.values(), key=sort_key)
        return snaps

    def download_snapshot(
        self, archive_url: str, path: str, timestamp: str
    ) -> Dict[str, Any]:
        """Download one snapshot, clean to text, extract images/taglines, and return a record."""
        try:
            r = self.session.get(archive_url, timeout=60)
            r.raise_for_status()
            html = r.text
            text = html_to_text(archive_url, html)

            # Parse HTML for image and tagline extraction
            soup = BeautifulSoup(html, "lxml")

            # Extract hero images and compute hashes
            hero_images = extract_hero_images(soup, self.base_domain)
            image_hashes = []
            for img_url in hero_images[
                :3
            ]:  # Limit to first 3 images to avoid too many requests
                img_hash = get_image_hash(img_url, self.session)
                if img_hash:
                    image_hashes.append({"url": img_url, "hash": img_hash})

            # Extract taglines
            taglines = extract_taglines(soup)

            # Persist
            slug = safe_slug(path)
            out_dir = OUTPUT_DIR / slug
            out_dir.mkdir(parents=True, exist_ok=True)

            html_fp = out_dir / f"{timestamp}.html"
            txt_fp = out_dir / f"{timestamp}.txt"
            with open(html_fp, "w", encoding="utf-8") as f:
                f.write(html)
            with open(txt_fp, "w", encoding="utf-8") as f:
                f.write(text)

            return {
                "path": path,
                "slug": slug,
                "timestamp": timestamp,
                "archive_url": archive_url,
                "html_path": str(html_fp),
                "text_path": str(txt_fp),
                "text_hash": hash_text(text),
                "chars": len(text),
                "hero_images": hero_images,
                "image_hashes": image_hashes,
                "taglines": taglines,
            }
        except Exception as e:
            logging.error(f"Failed to download snapshot {archive_url}: {e}")
            raise

    def run(self, paths: List[str]) -> pd.DataFrame:
        """Main entry: list→sample→download→index for all paths."""
        rows = []
        for path in paths:
            logging.info(f"[Collect] {path}")
            snaps = self.list_snapshots(path)
            snaps = self.sample_snapshots(snaps)
            for ts, url in snaps:
                try:
                    rec = self.download_snapshot(url, path, ts)
                    rows.append(rec)
                    time.sleep(REQUEST_SLEEP)
                except Exception as e:
                    logging.warning(f"  ! Skip {path}@{ts}: {e}")
        df = pd.DataFrame(rows)
        if not df.empty:
            df["dt_utc"] = pd.to_datetime(
                df["timestamp"], format="%Y%m%d%H%M%S", utc=True
            )
            df["dt_local"] = df["dt_utc"].dt.tz_convert(TIMEZONE)
        index_fp = OUTPUT_DIR / "index.csv"
        df.to_csv(index_fp, index=False, encoding="utf-8")
        logging.info(f"[Collect] Wrote index: {index_fp} ({len(df)} rows)")
        return df


# ------------------------ DiffAnalyzer ------------------------


@dataclass
class DiffAnalyzer:
    index_df: pd.DataFrame
    keywords: Dict[str, List[str]] = field(default_factory=lambda: KEYWORDS)
    significant_change: float = SIGNIFICANT_CHANGE

    def _load_text(self, fp: str) -> str:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()

    def _compare_image_hashes(
        self, hashes_a: List[Dict], hashes_b: List[Dict]
    ) -> Dict[str, Any]:
        """Compare image hashes between two snapshots."""
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            return {"image_changes": 0, "image_details": "PIL/imagehash not available"}

        if not hashes_a or not hashes_b:
            return {"image_changes": 0, "image_details": "No images to compare"}

        # Find matching images by URL or position
        changes = 0
        details = []

        # Simple approach: compare hashes by position
        for i, (hash_a, hash_b) in enumerate(zip(hashes_a, hashes_b)):
            try:
                # Convert string hashes back to imagehash objects for comparison
                if hash_a.get("hash") and hash_b.get("hash"):
                    # Calculate Hamming distance between hashes
                    hamming_dist = bin(
                        int(hash_a["hash"], 16) ^ int(hash_b["hash"], 16)
                    ).count("1")
                    if hamming_dist > IMAGE_HASH_THRESHOLD:
                        changes += 1
                        details.append(
                            f"Image {i}: {hash_a['url']} -> {hash_b['url']} (distance: {hamming_dist})"
                        )
            except Exception as e:
                details.append(f"Image {i}: Error comparing hashes - {e}")

        return {
            "image_changes": changes,
            "image_details": (
                "; ".join(details) if details else "No significant changes"
            ),
        }

    def _compare_taglines(
        self, taglines_a: Dict[str, str], taglines_b: Dict[str, str]
    ) -> Dict[str, Any]:
        """Compare taglines between two snapshots."""
        if not taglines_a and not taglines_b:
            return {"tagline_changes": 0, "tagline_details": "No taglines found"}

        changes = 0
        details = []

        # Find added, removed, and changed taglines
        all_keys = set(taglines_a.keys()) | set(taglines_b.keys())

        for key in all_keys:
            val_a = taglines_a.get(key, "")
            val_b = taglines_b.get(key, "")

            if val_a and not val_b:
                changes += 1
                details.append(f"Removed {key}: '{val_a[:50]}...'")
            elif not val_a and val_b:
                changes += 1
                details.append(f"Added {key}: '{val_b[:50]}...'")
            elif val_a != val_b:
                changes += 1
                details.append(f"Changed {key}: '{val_a[:30]}...' -> '{val_b[:30]}...'")

        return {
            "tagline_changes": changes,
            "tagline_details": "; ".join(details) if details else "No changes",
        }

    def _keyword_counts(self, text: str) -> Dict[str, int]:
        text_l = text.lower()
        counts = {}
        for bucket, words in self.keywords.items():
            c = 0
            for w in words:
                c += len(re.findall(r"\b" + re.escape(w.lower()) + r"\b", text_l))
            counts[bucket] = c
        return counts

    def _pairwise(self, group: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        For each path, compare consecutive snapshots chronologically:
        - cosine distance
        - unified diff (short)
        - keyword deltas
        """
        out = []
        g = group.sort_values("dt_utc").reset_index(drop=True)
        for i in range(1, len(g)):
            a = g.loc[i - 1]
            b = g.loc[i]
            A = self._load_text(a["text_path"])
            B = self._load_text(b["text_path"])

            dist = cosine_distance(A, B)
            kwA = self._keyword_counts(A)
            kwB = self._keyword_counts(B)
            kw_delta = {k: kwB[k] - kwA[k] for k in kwA}

            # small unified diff (first N lines)
            diff_lines = list(
                difflib.unified_diff(
                    A.splitlines(),
                    B.splitlines(),
                    fromfile=f"{a['slug']}@{a['timestamp']}",
                    tofile=f"{b['slug']}@{b['timestamp']}",
                    n=2,
                )
            )
            # keep a short snippet (avoid massive reports)
            snippet = "\n".join(diff_lines[:200])

            # Compare image hashes if available
            image_comparison = {}
            if "image_hashes" in a and "image_hashes" in b:
                try:
                    hashes_a = (
                        a["image_hashes"] if isinstance(a["image_hashes"], list) else []
                    )
                    hashes_b = (
                        b["image_hashes"] if isinstance(b["image_hashes"], list) else []
                    )
                    image_comparison = self._compare_image_hashes(hashes_a, hashes_b)
                except Exception as e:
                    image_comparison = {
                        "image_changes": 0,
                        "image_details": f"Error: {e}",
                    }

            # Compare taglines if available
            tagline_comparison = {}
            if "taglines" in a and "taglines" in b:
                try:
                    taglines_a = (
                        a["taglines"] if isinstance(a["taglines"], dict) else {}
                    )
                    taglines_b = (
                        b["taglines"] if isinstance(b["taglines"], dict) else {}
                    )
                    tagline_comparison = self._compare_taglines(taglines_a, taglines_b)
                except Exception as e:
                    tagline_comparison = {
                        "tagline_changes": 0,
                        "tagline_details": f"Error: {e}",
                    }

            result = {
                "path": a["path"],
                "slug": a["slug"],
                "from_ts": a["timestamp"],
                "to_ts": b["timestamp"],
                "from_url": a["archive_url"],
                "to_url": b["archive_url"],
                "from_local": a["dt_local"],
                "to_local": b["dt_local"],
                "cosine_distance": round(dist, 4),
                **{f"delta_{k}": v for k, v in kw_delta.items()},
                "diff_snippet": snippet,
            }

            # Add image and tagline comparison data
            result.update(image_comparison)
            result.update(tagline_comparison)

            out.append(result)
        return out

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          - diffs_df: row per consecutive pair with distance + keyword deltas + diff snippet
          - changes_df: subset where distance exceeds threshold (significant changes)
        Also writes CSVs and a Markdown summary.
        """
        diffs = []
        for slug, grp in self.index_df.groupby("slug"):
            diffs.extend(self._pairwise(grp))
        diffs_df = pd.DataFrame(diffs)
        diffs_fp = REPORT_DIR / "wayback_diffs.csv"
        diffs_df.to_csv(diffs_fp, index=False, encoding="utf-8")

        # Significant changes only
        changes_df = diffs_df[
            diffs_df["cosine_distance"] >= self.significant_change
        ].copy()
        changes_fp = REPORT_DIR / "wayback_changes_significant.csv"
        changes_df.to_csv(changes_fp, index=False, encoding="utf-8")

        # Keyword trend summary by path & year
        trend_rows = []
        for slug, grp in self.index_df.groupby("slug"):
            for _, r in grp.sort_values("dt_utc").iterrows():
                t = self._load_text(r["text_path"])
                counts = self._keyword_counts(t)
                counts.update(
                    {
                        "slug": slug,
                        "path": r["path"],
                        "timestamp": r["timestamp"],
                        "year": int(str(r["timestamp"])[:4]),
                        "dt_local": r["dt_local"],
                    }
                )
                trend_rows.append(counts)
        trend_df = pd.DataFrame(trend_rows)
        trend_fp = REPORT_DIR / "keyword_trends.csv"
        trend_df.to_csv(trend_fp, index=False, encoding="utf-8")

        # Markdown summary (compact)
        md = []
        md.append("# Wayback Brand Evolution — Change Log\n")
        md.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
        md.append(
            f"\n**Threshold for significant change:** cosine distance ≥ {self.significant_change}\n"
        )
        for _, r in changes_df.sort_values(["slug", "from_ts"]).iterrows():
            md.append(
                f"\n## {r['slug']} — {r['from_ts']} → {r['to_ts']} (distance {r['cosine_distance']})"
            )
            md.append(f"- From: {r['from_url']}")
            md.append(f"- To:   {r['to_url']}")
            deltas = {
                k.replace("delta_", ""): int(v)
                for k, v in r.items()
                if k.startswith("delta_")
            }
            if deltas:
                top = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[
                    :5
                ]
                md.append(
                    "- Top keyword deltas: "
                    + ", ".join([f"{k}:{v:+d}" for k, v in top])
                )
            md.append(
                "\n<details><summary>Diff snippet</summary>\n\n```\n"
                + (r["diff_snippet"] or "")
                + "\n```\n</details>\n"
            )
        md_fp = REPORT_DIR / "wayback_changes_summary.md"
        with open(md_fp, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        logging.info(f"[Analyze] diffs: {diffs_fp}")
        logging.info(f"[Analyze] significant changes: {changes_fp}")
        logging.info(f"[Analyze] keyword trends: {trend_fp}")
        logging.info(f"[Analyze] markdown summary: {md_fp}")
        return diffs_df, changes_df


# ------------------------ Main ------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_dirs()

    collector = WaybackCollector()
    index_df = collector.run(CANONICAL_PATHS)
    if index_df.empty:
        logging.warning("No snapshots collected. Check network or paths.")
        return

    analyzer = DiffAnalyzer(index_df=index_df)
    analyzer.run()


if __name__ == "__main__":
    main()
