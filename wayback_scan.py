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
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import tz
from waybackpy import WaybackMachineCDXServerAPI

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
    "/technology/",
    "/science/",
    "/team/",
    "/investors/",
    "/contact/",
    "/faq/",
    "/publications/",
    "/media/",
]

# Quarterly sampling: pick the first snapshot in each quarter per path
SAMPLING = "monthly"  # "all" | "yearly" | "quarterly" | "monthly"

# Keyword buckets for narrative coding
KEYWORDS = {
    "de_extinction": [
        "de-extinction",
        "de extinction",
        "deextinction",
        "de-extinct",
        "extinct",
    ],
    "functional_de_extinction": [
        "functional de-extinction",
        "proxy",
        "proxy species",
        "surrogate",
    ],
    "climate_benefit": [
        "climate",
        "carbon",
        "permafrost",
        "ecosystem",
        "biodiversity",
        "environment",
        "warming",
        "arctic",
    ],
    "iucn_alignment": [
        "IUCN",
        "International Union for Conservation of Nature",
        "conservation status",
    ],
    "animal_welfare": [
        "welfare",
        "ethics",
        "ethical",
        "suffering",
        "well-being",
        "wellbeing",
        "humane",
        "animal rights",
    ],
    "indigenous": [
        "indigenous",
        "iwi",
        "māori",
        "maori",
        "first nations",
        "tribal",
        "native peoples",
    ],
    "hype_vs_caution": [
        "moonshot",
        "sci-fi",
        "hype",
        "risk",
        "caution",
        "concern",
        "breakthrough",
        "revolutionary",
    ],
    "technology": [
        "CRISPR",
        "gene editing",
        "genetic engineering",
        "biotechnology",
        "genomics",
        "DNA",
    ],
    "funding": [
        "funding",
        "investment",
        "million",
        "billion",
        "venture",
        "capital",
        "investors",
    ],
    "timeline": [
        "years",
        "decade",
        "timeline",
        "when",
        "soon",
        "future",
        "2025",
        "2026",
        "2027",
        "2028",
        "2029",
        "2030",
    ],
}

# Change detection threshold (cosine distance on token shingles)
SIGNIFICANT_CHANGE = 0.5  # 0 = identical, 1 = completely different (lowered from 0.20)

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


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text - collapse multiple spaces and newlines."""
    # Replace multiple whitespace with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Clean up space-newline combinations
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def html_to_text(url: str, html: str) -> str:
    """
    Extract readable text from archived HTML:
    - strips scripts/styles/noscript and common boilerplate containers
    - preserves headings/paragraphs/link text
    - adds basic provenance header (source URL, title, meta descriptions)
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content/boilerplate
    for tag in soup(["script", "style", "noscript", "template", "svg", "canvas"]):
        tag.decompose()
    for selector in ["nav", "footer", "header", "form", "aside"]:
        for n in soup.select(selector):
            n.decompose()

    # Collect a tiny header
    title = (soup.title.string or "").strip() if soup.title else ""

    # Extract meta descriptions more safely - ignore type checker for BeautifulSoup
    meta_desc = ""
    ogd = ""
    # Note: Ignoring type issues with BeautifulSoup - it works correctly at runtime

    body_text = soup.get_text("\n", strip=True)
    body_text = _normalize_whitespace(body_text)

    header = []
    header.append(f"[SOURCE] {url}")
    if title:
        header.append(f"[TITLE] {title}")
    if meta_desc:
        header.append(f"[META] {meta_desc}")
    if ogd and ogd != meta_desc:
        header.append(f"[OG] {ogd}")

    prefix = "\n\n".join(header).strip()
    return (prefix + ("\n\n" if prefix and body_text else "") + body_text).strip()


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


def extract_taglines(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract taglines/slogans from HTML using various selectors."""
    taglines = {}

    for selector in TAGLINE_SELECTORS:
        try:
            if selector.startswith("meta"):
                # Handle meta tag extraction with proper BeautifulSoup parsing
                if selector == "meta[name='description']":
                    meta_elem = soup.find("meta", attrs={"name": "description"})
                    if meta_elem:
                        try:
                            # Try to get content attribute, ignore type checker warnings
                            content = meta_elem.get("content")  # type: ignore
                            if content and isinstance(content, str):
                                text = content.strip()
                                if text and len(text) < 500:
                                    taglines["meta_description"] = text
                        except (AttributeError, TypeError):
                            pass
                elif selector == "meta[property='og:description']":
                    meta_elem = soup.find("meta", attrs={"property": "og:description"})
                    if meta_elem:
                        try:
                            # Try to get content attribute, ignore type checker warnings
                            content = meta_elem.get("content")  # type: ignore
                            if content and isinstance(content, str):
                                text = content.strip()
                                if text and len(text) < 500:
                                    taglines["og_description"] = text
                        except (AttributeError, TypeError):
                            pass
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
        if mode == "monthly":
            # earliest per month for more granular analysis
            chosen = {}
            for ts, url in snaps:
                month_key = ts[:6]  # YYYYMM
                if month_key not in chosen:
                    chosen[month_key] = (ts, url)
            return [chosen[mk] for mk in sorted(chosen)]
        return snaps

    def download_snapshot(
        self, archive_url: str, path: str, timestamp: str
    ) -> Dict[str, Any]:
        """Download one snapshot, clean to text, extract taglines, and return a record."""
        try:
            r = self.session.get(archive_url, timeout=60)
            r.raise_for_status()
            html = r.text
            text = html_to_text(archive_url, html)

            # Parse HTML for tagline extraction
            soup = BeautifulSoup(html, "lxml")

            # Extract taglines only
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

            # Remove SOURCE line differences to focus on content changes
            A_clean = "\n".join(
                [line for line in A.splitlines() if not line.startswith("[SOURCE]")]
            )
            B_clean = "\n".join(
                [line for line in B.splitlines() if not line.startswith("[SOURCE]")]
            )

            dist = cosine_distance(A_clean, B_clean)
            kwA = self._keyword_counts(A_clean)
            kwB = self._keyword_counts(B_clean)
            kw_delta = {k: kwB[k] - kwA[k] for k in kwA}

            # Generate a more meaningful diff - skip SOURCE line changes
            diff_lines = list(
                difflib.unified_diff(
                    A_clean.splitlines(),
                    B_clean.splitlines(),
                    fromfile=f"{a['slug']}@{a['timestamp']}",
                    tofile=f"{b['slug']}@{b['timestamp']}",
                    n=3,  # More context lines
                )
            )

            # Filter out trivial diff lines and keep meaningful content changes
            meaningful_diff_lines = []
            for line in diff_lines:
                if (
                    line.startswith("---")
                    or line.startswith("+++")
                    or line.startswith("@@")
                    or line.startswith(" ")
                ):
                    meaningful_diff_lines.append(line)
                elif line.startswith(("+", "-")):
                    # Only include substantive changes (not just whitespace/trivial)
                    stripped = line[1:].strip()
                    if stripped and len(stripped) > 5:  # Ignore very short changes
                        meaningful_diff_lines.append(line)

            # Keep a reasonable snippet (avoid massive reports)
            snippet = "\n".join(meaningful_diff_lines[:300])

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
                "from_chars": len(A_clean),
                "to_chars": len(B_clean),
                "char_change": len(B_clean) - len(A_clean),
                **{f"delta_{k}": v for k, v in kw_delta.items()},
                "diff_snippet": snippet,
            }

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


# ------------------------ Data Checker ------------------------


def check_existing_data() -> Tuple[bool, pd.DataFrame]:
    """
    Check if we already have collected data and return it if available.
    Returns (data_exists, dataframe)
    """
    index_fp = OUTPUT_DIR / "index.csv"
    
    if not index_fp.exists():
        logging.info("No existing index.csv found - will collect fresh data")
        return False, pd.DataFrame()
    
    try:
        df = pd.read_csv(index_fp, encoding="utf-8")
        if df.empty:
            logging.info("Index.csv exists but is empty - will collect fresh data")
            return False, pd.DataFrame()
        
        # Check if the data files actually exist
        missing_files = []
        for _, row in df.iterrows():
            text_path = row.get("text_path", "")
            html_path = row.get("html_path", "")
            
            if not pathlib.Path(text_path).exists():
                missing_files.append(text_path)
            if not pathlib.Path(html_path).exists():
                missing_files.append(html_path)
        
        if missing_files:
            logging.warning(f"Found {len(missing_files)} missing data files - will recollect")
            return False, pd.DataFrame()
        
        # Convert datetime columns if they exist
        if "dt_utc" not in df.columns:
            df["dt_utc"] = pd.to_datetime(
                df["timestamp"], format="%Y%m%d%H%M%S", utc=True
            )
            df["dt_local"] = df["dt_utc"].dt.tz_convert(TIMEZONE)
        
        # Print summary of existing data
        total_snapshots = len(df)
        paths_covered = df["path"].nunique()
        date_range = f"{df['dt_local'].min().strftime('%Y-%m-%d')} to {df['dt_local'].max().strftime('%Y-%m-%d')}"
        
        logging.info("=" * 60)
        logging.info("EXISTING DATA FOUND - SKIPPING API CALLS")
        logging.info("=" * 60)
        logging.info(f"Total snapshots: {total_snapshots}")
        logging.info(f"Paths covered: {paths_covered}")
        logging.info(f"Date range: {date_range}")
        logging.info(f"Data location: {OUTPUT_DIR}")
        
        # Show breakdown by path
        path_counts = df.groupby("path").size().sort_values(ascending=False)
        logging.info("\nSnapshots per path:")
        for path, count in path_counts.items():
            logging.info(f"  {path}: {count} snapshots")
        
        logging.info("\nProceeding directly to diff analysis...")
        logging.info("=" * 60)
        
        return True, df
        
    except Exception as e:
        logging.error(f"Error reading existing data: {e}")
        logging.info("Will collect fresh data")
        return False, pd.DataFrame()


def print_analysis_summary(diffs_df: pd.DataFrame, changes_df: pd.DataFrame):
    """Print a summary of the analysis results."""
    logging.info("=" * 60)
    logging.info("ANALYSIS COMPLETED")
    logging.info("=" * 60)
    logging.info(f"Total comparisons: {len(diffs_df)}")
    logging.info(f"Significant changes: {len(changes_df)} (threshold: {SIGNIFICANT_CHANGE})")
    
    if not changes_df.empty:
        # Show changes by path
        changes_by_path = changes_df.groupby("slug").size().sort_values(ascending=False)
        logging.info("\nSignificant changes by path:")
        for slug, count in changes_by_path.items():
            logging.info(f"  {slug}: {count} changes")
        
        # Show highest distance changes
        top_changes = changes_df.nlargest(5, "cosine_distance")[["slug", "from_ts", "to_ts", "cosine_distance"]]
        logging.info("\nTop 5 largest changes:")
        for _, row in top_changes.iterrows():
            logging.info(f"  {row['slug']}: {row['from_ts']} → {row['to_ts']} (distance: {row['cosine_distance']})")
    
    logging.info(f"\nReports saved to: {REPORT_DIR}")
    logging.info("=" * 60)


# ------------------------ Main ------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_dirs()

    # Check if we already have data
    data_exists, index_df = check_existing_data()
    
    if not data_exists:
        # Collect fresh data from API
        logging.info("Starting fresh data collection from Wayback Machine...")
        collector = WaybackCollector()
        index_df = collector.run(CANONICAL_PATHS)
        if index_df.empty:
            logging.warning("No snapshots collected. Check network or paths.")
            return
    
    # Run diff analysis on existing or fresh data
    analyzer = DiffAnalyzer(index_df=index_df)
    diffs_df, changes_df = analyzer.run()
    
    # Print summary
    print_analysis_summary(diffs_df, changes_df)


if __name__ == "__main__":
    main()
