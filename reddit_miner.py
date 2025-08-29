from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

import pandas as pd
import praw
from dotenv import load_dotenv
from praw.models import Redditor, Subreddit

# ---------------------------- Configuration ----------------------------

ENV_FILE = "environment.env"


@dataclass(frozen=True)
class RedditAuth:
    client_id: str
    client_secret: str
    user_agent: str

    @staticmethod
    def from_env(env_file: str = ENV_FILE) -> "RedditAuth":
        load_dotenv(env_file)
        cid = os.getenv("REDDIT_CLIENT_ID")
        csec = os.getenv("REDDIT_CLIENT_SECRET")
        ua = os.getenv("REDDIT_USER_AGENT")
        if not all([cid, csec, ua]):
            raise SystemExit(
                "Missing env vars. Ensure REDDIT_CLIENT_ID/SECRET/USER_AGENT in environment.env"
            )
        return RedditAuth(cast(str, cid), cast(str, csec), cast(str, ua))


@dataclass
class MinerOptions:
    limit: Optional[int] = None  # None = all available
    include_comments: bool = True
    polite_sleep: float = (
        0.0  # seconds between items to be extra polite (PRAW already rate-limits)
    )
    # Defaults used when limit is not specified
    default_post_limit: int = 1000
    default_comment_limit: int = 5000


class RedditClient:
    def __init__(self, auth: RedditAuth):
        self._reddit = praw.Reddit(
            client_id=auth.client_id,
            client_secret=auth.client_secret,
            user_agent=auth.user_agent,
        )

    @property
    def reddit(self) -> praw.Reddit:
        return self._reddit


# ---------------------------- Miner Interfaces ----------------------------


class BaseMiner:
    def records(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError


class UserMiner(BaseMiner):
    def __init__(self, reddit: praw.Reddit, username: str, options: MinerOptions):
        self.reddit = reddit
        self.username = username
        self.options = options
        self._user: Redditor = self.reddit.redditor(username)

    def _gen_posts(self) -> Iterable[Dict[str, Any]]:
        effective_limit = (
            self.options.limit
            if self.options.limit is not None
            else self.options.default_post_limit
        )
        kwargs = {"limit": effective_limit}
        count = 0
        start = time.time()
        last_print = start
        for s in self._user.submissions.new(**kwargs):
            yield {
                "type": "post",
                "id": s.id,
                "created_utc": getattr(s, "created_utc", None),
                "subreddit": s.subreddit.display_name,
                "title": getattr(s, "title", None),
                "selftext": getattr(s, "selftext", None),
                "url": getattr(s, "url", None),
                "score": getattr(s, "score", None),
                "num_comments": getattr(s, "num_comments", None),
                "permalink": "https://www.reddit.com" + getattr(s, "permalink", ""),
            }
            count += 1
            now = time.time()
            if (now - last_print) >= 2 or (count % 25 == 0) or count == effective_limit:
                rate = count / max(now - start, 1e-6)
                remaining = max(effective_limit - count, 0)
                eta = int(remaining / rate) if rate > 0 else 0
                hrs, rem = divmod(eta, 3600)
                mins, secs = divmod(rem, 60)
                eta_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
                logging.info(
                    f"UserMiner posts progress: {count}/{effective_limit} (ETA {eta_str})"
                )
                last_print = now
            if self.options.polite_sleep:
                time.sleep(self.options.polite_sleep)

    def _gen_comments(self) -> Iterable[Dict[str, Any]]:
        effective_limit = (
            self.options.limit
            if self.options.limit is not None
            else self.options.default_comment_limit
        )
        kwargs = {"limit": effective_limit}
        count = 0
        start = time.time()
        last_print = start
        for c in self._user.comments.new(**kwargs):
            yield {
                "type": "comment",
                "id": c.id,
                "created_utc": getattr(c, "created_utc", None),
                "subreddit": c.subreddit.display_name,
                "link_id": getattr(c, "link_id", None),
                "link_title": getattr(getattr(c, "submission", None), "title", None),
                "body": getattr(c, "body", None),
                "score": getattr(c, "score", None),
                "parent_id": getattr(c, "parent_id", None),
                "permalink": "https://www.reddit.com" + getattr(c, "permalink", ""),
            }
            count += 1
            now = time.time()
            if (now - last_print) >= 2 or (count % 50 == 0) or count == effective_limit:
                rate = count / max(now - start, 1e-6)
                remaining = max(effective_limit - count, 0)
                eta = int(remaining / rate) if rate > 0 else 0
                hrs, rem = divmod(eta, 3600)
                mins, secs = divmod(rem, 60)
                eta_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
                logging.info(
                    f"UserMiner comments progress: {count}/{effective_limit} (ETA {eta_str})"
                )
                last_print = now
            if self.options.polite_sleep:
                time.sleep(self.options.polite_sleep)

    def records(self) -> Iterable[Dict[str, Any]]:
        yield from self._gen_posts()
        if self.options.include_comments:
            yield from self._gen_comments()


class SubredditMiner(BaseMiner):
    """Collect submissions (and optionally recent comments) from a subreddit."""

    def __init__(self, reddit: praw.Reddit, subreddit_name: str, options: MinerOptions):
        self.reddit = reddit
        self.subreddit_name = subreddit_name
        self.options = options
        self._sub: Subreddit = self.reddit.subreddit(subreddit_name)

    def _gen_posts(self) -> Iterable[Dict[str, Any]]:
        effective_limit = (
            self.options.limit
            if self.options.limit is not None
            else self.options.default_post_limit
        )
        kwargs = {"limit": effective_limit}
        count = 0
        start = time.time()
        last_print = start
        for s in self._sub.new(**kwargs):
            yield {
                "type": "post",
                "id": s.id,
                "created_utc": getattr(s, "created_utc", None),
                "subreddit": s.subreddit.display_name,
                "title": getattr(s, "title", None),
                "selftext": getattr(s, "selftext", None),
                "url": getattr(s, "url", None),
                "score": getattr(s, "score", None),
                "num_comments": getattr(s, "num_comments", None),
                "permalink": "https://www.reddit.com" + getattr(s, "permalink", ""),
            }
            count += 1
            now = time.time()
            if (now - last_print) >= 2 or (count % 25 == 0) or count == effective_limit:
                rate = count / max(now - start, 1e-6)
                remaining = max(effective_limit - count, 0)
                eta = int(remaining / rate) if rate > 0 else 0
                hrs, rem = divmod(eta, 3600)
                mins, secs = divmod(rem, 60)
                eta_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
                logging.info(
                    f"SubredditMiner posts progress: {count}/{effective_limit} (ETA {eta_str})"
                )
                last_print = now
            if self.options.polite_sleep:
                time.sleep(self.options.polite_sleep)

    def _gen_comments(self) -> Iterable[Dict[str, Any]]:
        # Subreddit.comments() streams latest comments across the subreddit.
        effective_limit = (
            self.options.limit
            if self.options.limit is not None
            else self.options.default_comment_limit
        )
        kwargs = {"limit": effective_limit}
        count = 0
        start = time.time()
        last_print = start
        for c in self._sub.comments(**kwargs):
            yield {
                "type": "comment",
                "id": c.id,
                "created_utc": getattr(c, "created_utc", None),
                "subreddit": c.subreddit.display_name,
                "link_id": getattr(c, "link_id", None),
                "link_title": getattr(getattr(c, "submission", None), "title", None),
                "body": getattr(c, "body", None),
                "score": getattr(c, "score", None),
                "parent_id": getattr(c, "parent_id", None),
                "permalink": "https://www.reddit.com" + getattr(c, "permalink", ""),
            }
            count += 1
            now = time.time()
            if (now - last_print) >= 2 or (count % 50 == 0) or count == effective_limit:
                rate = count / max(now - start, 1e-6)
                remaining = max(effective_limit - count, 0)
                eta = int(remaining / rate) if rate > 0 else 0
                hrs, rem = divmod(eta, 3600)
                mins, secs = divmod(rem, 60)
                eta_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
                logging.info(
                    f"SubredditMiner comments progress: {count}/{effective_limit} (ETA {eta_str})"
                )
                last_print = now
            if self.options.polite_sleep:
                time.sleep(self.options.polite_sleep)

    def records(self) -> Iterable[Dict[str, Any]]:
        yield from self._gen_posts()
        if self.options.include_comments:
            yield from self._gen_comments()


# ---------------------------- Processing & Persistence (Excel) ----------------------------


class DataProcessor:
    """Process mined records and write a consolidated Excel workbook.

    Sheets:
    - Posts
    - Comments
    - Comments+Posts (joined via parent_id/link_id)
    """

    EXPECTED_COLS = [
        "type",
        "id",
        "created_utc",
        "subreddit",
        "title",
        "selftext",
        "url",
        "score",
        "num_comments",
        "permalink",
        "link_id",
        "body",
        "parent_id",
        "link_title",
    ]

    @staticmethod
    def _extract_post_id_from_t(idval: Optional[str]) -> Optional[str]:
        if not idval:
            return None
        if isinstance(idval, str) and idval.startswith("t3_"):
            return idval.split("_", 1)[1]
        return None

    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self.df = pd.DataFrame(list(rows))
        if self.df.empty:
            logging.warning("No rows collected; workbook will contain empty sheets.")
        for col in self.EXPECTED_COLS:
            if col not in self.df.columns:
                self.df[col] = None

    def _add_time_columns(self) -> None:
        # Convert epoch seconds to UTC then to Europe/Zurich, and drop timezone for Excel
        self.df["created_utc"] = pd.to_numeric(self.df["created_utc"], errors="coerce")
        dt_utc = pd.to_datetime(self.df["created_utc"], unit="s", utc=True)
        # Store tz-naive UTC for Excel compatibility
        self.df["created_dt_utc"] = dt_utc.dt.tz_localize(None)
        try:
            # Convert from original UTC (dt_utc) to Europe/Zurich and then make tz-naive
            created_dt_ch = dt_utc.dt.tz_convert("Europe/Zurich")
            self.df["created_dt_ch"] = created_dt_ch.dt.tz_localize(None)
        except Exception as e:
            logging.warning(f"Timezone conversion failed, using UTC (naive): {e}")
            self.df["created_dt_ch"] = self.df["created_dt_utc"]

    def _split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        posts = self.df[self.df["type"] == "post"].copy()
        comments = self.df[self.df["type"] == "comment"].copy()
        if not posts.empty:
            posts = posts.sort_values("created_utc")
        if not comments.empty:
            comments = comments.sort_values("created_utc")
        return posts, comments

    def _add_engagement(self, posts: pd.DataFrame) -> pd.DataFrame:
        if posts.empty:
            posts["engagement_score"] = pd.Series(dtype=float)
            return posts
        posts["score"] = pd.to_numeric(posts["score"], errors="coerce")
        posts["num_comments"] = pd.to_numeric(posts["num_comments"], errors="coerce")
        posts["engagement_score"] = (
            posts[["score", "num_comments"]].fillna(0).sum(axis=1)
        )
        return posts

    def _derive_post_ids(
        self, posts: pd.DataFrame, comments: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        posts["post_id"] = (
            posts["id"].astype(str) if not posts.empty else pd.Series(dtype=str)
        )
        if not comments.empty:
            pid_from_parent = (
                comments["parent_id"].astype(str).apply(self._extract_post_id_from_t)
            )
            pid_from_link = (
                comments["link_id"].astype(str).apply(self._extract_post_id_from_t)
            )
            comments["post_id"] = pid_from_parent.where(
                pid_from_parent.notna(), pid_from_link
            )
        else:
            comments["post_id"] = pd.Series(dtype=str)
        return posts, comments

    def _join(self, posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
        join_cols = ["post_id", "title", "permalink", "subreddit", "engagement_score"]
        if posts.empty:
            posts_join = pd.DataFrame(
                columns=[
                    "post_id",
                    "post_title",
                    "post_permalink",
                    "post_subreddit",
                    "post_engagement_score",
                ]
            )
        else:
            posts_join = posts[join_cols].rename(
                columns={
                    "title": "post_title",
                    "permalink": "post_permalink",
                    "subreddit": "post_subreddit",
                    "engagement_score": "post_engagement_score",
                }
            )
        return (
            comments.merge(posts_join, on="post_id", how="left")
            if not comments.empty
            else comments
        )

    def save_workbook(self, out_path: Path) -> Path:
        # Prepare time columns and split frames
        self._add_time_columns()
        posts, comments = self._split()
        posts = self._add_engagement(posts)
        posts, comments = self._derive_post_ids(posts, comments)
        comments_joined = self._join(posts, comments)

        # Ensure destination directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
            # Select relevant columns to avoid empty fields
            post_cols = [
                "post_id",
                "id",
                "created_utc",
                "created_dt_utc",
                "created_dt_ch",
                "subreddit",
                "title",
                "selftext",
                "url",
                "score",
                "num_comments",
                "engagement_score",
                "permalink",
            ]
            post_cols = [c for c in post_cols if c in posts.columns]
            comment_cols = [
                "id",
                "post_id",
                "created_utc",
                "created_dt_utc",
                "created_dt_ch",
                "subreddit",
                "link_title",
                "body",
                "score",
                "parent_id",
                "permalink",
            ]
            comment_cols = [c for c in comment_cols if c in comments.columns]

            posts[post_cols].to_excel(writer, sheet_name="Posts", index=False)
            comments[comment_cols].to_excel(writer, sheet_name="Comments", index=False)
            comments_joined.to_excel(writer, sheet_name="Comments+Posts", index=False)
        logging.info(
            f"Saved workbook → {out_path} (posts: {len(posts)}, comments: {len(comments)}, joined: {len(comments_joined)})"
        )
        return out_path


# ---------------------------- CLI ----------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Reddit miner (user or subreddit)")
    parser.add_argument(
        "mode", choices=["user", "subreddit"], help="Mining target type"
    )
    parser.add_argument("name", help="Username (without u/) or subreddit (without r/)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items per listing. If omitted, defaults to 1000 posts and 5000 comments.",
    )
    parser.add_argument(
        "--no-comments", action="store_true", help="Do not include comments"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional polite sleep between items (sec)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Excel path (.xlsx). If omitted, writes to reddit_data/r-<sub>.xlsx or u-<user>.xlsx",
    )
    parser.add_argument(
        "--env", type=str, default=ENV_FILE, help="Path to environment file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    auth = RedditAuth.from_env(args.env)
    client = RedditClient(auth)
    options = MinerOptions(
        limit=args.limit, include_comments=not args.no_comments, polite_sleep=args.sleep
    )

    project_root = Path(__file__).resolve().parent
    default_dir = project_root / "reddit_data"
    if args.mode == "user":
        miner: BaseMiner = UserMiner(client.reddit, args.name, options)
        default_name = f"u-{args.name}.xlsx"
    else:
        miner = SubredditMiner(client.reddit, args.name, options)
        default_name = f"r-{args.name}.xlsx"

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() != ".xlsx":
            out_path = out_path.with_suffix(".xlsx")
    else:
        out_path = default_dir / default_name

    logging.info(f"Read-only? {client.reddit.read_only}")
    processor = DataProcessor(miner.records())
    saved = processor.save_workbook(out_path)
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
