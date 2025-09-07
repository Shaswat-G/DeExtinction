from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import time
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup  # added
from waybackpy import WaybackMachineCDXServerAPI


def safe_slug(path: str) -> str:
    """Convert a URL path to a filesystem-friendly slug."""
    return re.sub(r"[^a-zA-Z0-9\-]+", "-", path.strip("/")).strip("-") or "home"


def list_snapshots(
    base_url: str, path: str, start: str, end: str, user_agent: str
) -> List[Tuple[str, str]]:
    """
    Return list of (timestamp, archive_url) for a given path between start/end.
    """
    url = base_url.rstrip("/") + path
    cdx = WaybackMachineCDXServerAPI(
        url,
        start_timestamp=start,
        end_timestamp=end,
        user_agent=user_agent,
    )
    return [(s.timestamp, s.archive_url) for s in cdx.snapshots()]


def download_snapshot(
    session: requests.Session, archive_url: str, timeout: int = 60
) -> bytes:
    """Fetch archived HTML bytes for a snapshot URL."""
    resp = session.get(archive_url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


# ---------------- Text extraction (simple, no images) ----------------


def _normalize_whitespace(text: str) -> str:
    # Collapse 3+ newlines, strip lines, remove trailing spaces
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    # Drop leading/trailing blank lines
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def extract_text_from_html_bytes(
    html_bytes: bytes, source_url: str | None = None
) -> str:
    """
    Extract readable text from archived HTML:
    - strips scripts/styles/noscript and common boilerplate containers
    - preserves headings/paragraphs/link text
    - adds basic provenance header (source URL, title, meta descriptions)
    """
    html = html_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content/boilerplate
    for tag in soup(["script", "style", "noscript", "template", "svg", "canvas"]):
        tag.decompose()
    for selector in ["nav", "footer", "header", "form", "aside"]:
        for n in soup.select(selector):
            n.decompose()

    # Collect a tiny header
    title = (soup.title.string or "").strip() if soup.title else ""
    meta_desc = ""
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        meta_desc = md["content"].strip()
    ogd = ""
    og = soup.find("meta", attrs={"property": "og:description"})
    if og and og.get("content"):
        ogd = og["content"].strip()

    body_text = soup.get_text("\n", strip=True)
    body_text = _normalize_whitespace(body_text)

    header = []
    if source_url:
        header.append(f"[SOURCE] {source_url}")
    if title:
        header.append(f"[TITLE] {title}")
    if meta_desc:
        header.append(f"[META] {meta_desc}")
    if ogd and ogd != meta_desc:
        header.append(f"[OG] {ogd}")

    prefix = "\n\n".join(header).strip()
    return (prefix + ("\n\n" if prefix and body_text else "") + body_text).strip()


def run(
    base_url: str,
    paths: List[str],
    start: str,
    end: str,
    out_dir: pathlib.Path,
    sleep: float,
    limit_per_path: int,
    user_agent: str,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    # path, timestamp, archive_url, html_path, text_path, text_chars
    index_rows: List[Tuple[str, str, str, str, str, int]] = []

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    for path in paths:
        print(
            f"\n[CDX] Listing snapshots for {base_url.rstrip('/') + path} between {start} and {end} ..."
        )
        try:
            snaps = list_snapshots(base_url, path, start, end, user_agent)
        except Exception as e:
            print(f"  ! Failed to query CDX for {path}: {e}")
            continue

        print(f"  Found {len(snaps)} snapshots")
        if not snaps:
            continue

        slug = safe_slug(path)
        path_dir = out_dir / slug
        path_dir.mkdir(parents=True, exist_ok=True)

        for i, (ts, archive_url) in enumerate(snaps[: max(0, limit_per_path)]):
            print(f"  [{i+1}/{min(len(snaps), limit_per_path)}] {ts} → {archive_url}")
            try:
                print("    Downloading HTML ...", end="", flush=True)
                html_bytes = download_snapshot(session, archive_url)
                html_fp = path_dir / f"{ts}.html"
                html_fp.write_bytes(html_bytes)
                print(f" saved {html_fp} ({len(html_bytes)} bytes)")

                # Extract text and save alongside HTML
                print("    Extracting text ...", end="", flush=True)
                text = extract_text_from_html_bytes(html_bytes, source_url=archive_url)
                txt_fp = path_dir / f"{ts}.txt"
                txt_fp.write_text(text, encoding="utf-8")
                print(f" saved {txt_fp} ({len(text)} chars)")
                index_rows.append(
                    (path, ts, archive_url, str(html_fp), str(txt_fp), len(text))
                )
            except Exception as e:
                print(f"\n    ! Error downloading/processing {archive_url}: {e}")
            if sleep > 0:
                time.sleep(sleep)

    # Write index.csv for quick reference
    index_fp = out_dir / "index.csv"
    with index_fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["path", "timestamp", "archive_url", "html_file", "text_file", "text_chars"]
        )
        w.writerows(index_rows)
    print(f"\n[Done] Wrote index: {index_fp} ({len(index_rows)} rows)")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Basic Wayback snapshot scraper using CDX server API"
    )
    p.add_argument(
        "--base",
        type=str,
        default="https://colossal.com",
        help="Base URL (e.g., https://example.com)",
    )
    p.add_argument(
        "--paths",
        nargs="*",
        default=["/"],
        help="URL paths to scan (space-separated), default: /",
    )
    p.add_argument(
        "--start",
        type=str,
        default="20210101",
        help="Start timestamp (YYYYMMDD or YYYYMMDDhhmmss)",
    )
    p.add_argument(
        "--end",
        type=str,
        default="20251231",
        help="End timestamp (YYYYMMDD or YYYYMMDDhhmmss)",
    )
    p.add_argument(
        "--limit", type=int, default=5, help="Max snapshots per path to download"
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Polite delay between downloads (seconds)",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("data/wayback_basic"),
        help="Output directory",
    )
    p.add_argument(
        "--user-agent",
        type=str,
        default="WaybackBasicScanner/1.0 (+https://example.com)",
        help="HTTP User-Agent",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(
            base_url=args.base,
            paths=args.paths,
            start=args.start,
            end=args.end,
            out_dir=args.out,
            sleep=args.sleep,
            limit_per_path=args.limit,
            user_agent=args.user_agent,
        )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
