from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
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
            raise SystemExit("Missing env vars. Ensure REDDIT_CLIENT_ID/SECRET/USER_AGENT in environment.env")
        return RedditAuth(cast(str, cid), cast(str, csec), cast(str, ua))


@dataclass
class MinerOptions:
    limit: Optional[int] = None  # None = all available
    include_comments: bool = True
    polite_sleep: float = (0.0)  # seconds between items to be extra polite (PRAW already rate-limits)


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
        kwargs = {"limit": self.options.limit} if self.options.limit is not None else {}
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
            if self.options.polite_sleep:
                time.sleep(self.options.polite_sleep)

    def _gen_comments(self) -> Iterable[Dict[str, Any]]:
        kwargs = {"limit": self.options.limit} if self.options.limit is not None else {}
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
        kwargs = {"limit": self.options.limit} if self.options.limit is not None else {}
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
            if self.options.polite_sleep:
                time.sleep(self.options.polite_sleep)

    def _gen_comments(self) -> Iterable[Dict[str, Any]]:
        # Subreddit.comments() streams latest comments across the subreddit.
        kwargs = {"limit": self.options.limit} if self.options.limit is not None else {}
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
            if self.options.polite_sleep:
                time.sleep(self.options.polite_sleep)

    def records(self) -> Iterable[Dict[str, Any]]:
        yield from self._gen_posts()
        if self.options.include_comments:
            yield from self._gen_comments()


# ---------------------------- Persistence (Excel) ----------------------------

def _extract_post_id_from_t(idval: Optional[str]) -> Optional[str]:
    """Return the bare post id if the fullname references a submission (t3_), else None."""
    if not idval:
        return None
    if isinstance(idval, str) and idval.startswith("t3_"):
        return idval.split("_", 1)[1]
    return None


def save_excel(rows: Iterable[Dict[str, Any]], out_prefix: str) -> str:
    """Write an Excel workbook (.xlsx) with three sheets:
    - Posts: submissions with engagement = score + num_comments
    - Comments: raw comments
    - Comments+Posts: comments joined with parent post metadata via parent_id/link_id
    """
    all_rows = list(rows)
    if not all_rows:
        logging.warning("No rows collected; creating an empty workbook.")

    df = pd.DataFrame(all_rows)

    # Ensure expected columns exist
    expected = [
        "type","id","created_utc","subreddit","title","selftext","url","score",
        "num_comments","permalink","link_id","body","parent_id","link_title",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = None

    # Split posts and comments and sort
    posts = df[df["type"] == "post"].copy()
    comments = df[df["type"] == "comment"].copy()
    if not posts.empty:
        posts = posts.sort_values("created_utc")
    if not comments.empty:
        comments = comments.sort_values("created_utc")

    # Compute engagement for posts
    if not posts.empty:
        posts["score"] = pd.to_numeric(posts["score"], errors="coerce")
        posts["num_comments"] = pd.to_numeric(posts["num_comments"], errors="coerce")
        posts["engagement_score"] = posts[["score", "num_comments"]].fillna(0).sum(axis=1)

    # Derive post_id on posts and comments to enable join
    posts["post_id"] = posts["id"].astype(str) if not posts.empty else pd.Series(dtype=str)
    if not comments.empty:
        pid_from_parent = comments["parent_id"].astype(str).apply(_extract_post_id_from_t)
        pid_from_link = comments["link_id"].astype(str).apply(_extract_post_id_from_t)
        comments["post_id"] = pid_from_parent.where(pid_from_parent.notna(), pid_from_link)
    else:
        comments["post_id"] = pd.Series(dtype=str)

    # Build joined view (comments with parent post fields)
    join_cols = ["post_id", "title", "permalink", "subreddit", "engagement_score"]
    posts_join = posts[join_cols].rename(columns={
        "title": "post_title",
        "permalink": "post_permalink",
        "subreddit": "post_subreddit",
        "engagement_score": "post_engagement_score",
    }) if not posts.empty else pd.DataFrame(columns=[
        "post_id","post_title","post_permalink","post_subreddit","post_engagement_score"
    ])
    comments_joined = comments.merge(posts_join, on="post_id", how="left") if not comments.empty else comments

    # Write Excel workbook
    xlsx_path = f"{out_prefix}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        posts.to_excel(writer, sheet_name="Posts", index=False)
        comments.to_excel(writer, sheet_name="Comments", index=False)
        comments_joined.to_excel(writer, sheet_name="Comments+Posts", index=False)

    logging.info(
        f"Saved workbook → {xlsx_path} (posts: {len(posts)}, comments: {len(comments)}, joined: {len(comments_joined)})"
    )
    return xlsx_path


# ---------------------------- CLI ----------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Reddit miner (user or subreddit)")
    parser.add_argument("mode", choices=["user", "subreddit"], help="Mining target type")
    parser.add_argument("name", help="Username (without u/) or subreddit (without r/)")
    parser.add_argument("--limit", type=int, default=None, help="Max items per listing (None=all)")
    parser.add_argument("--no-comments", action="store_true", help="Do not include comments")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional polite sleep between items (sec)")
    parser.add_argument("--output", type=str, default=None, help="Output file prefix (without extension)")
    parser.add_argument("--env", type=str, default=ENV_FILE, help="Path to environment file")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s",)

    auth = RedditAuth.from_env(args.env)
    client = RedditClient(auth)
    options = MinerOptions(limit=args.limit, include_comments=not args.no_comments, polite_sleep=args.sleep)

    if args.mode == "user":
        miner: BaseMiner = UserMiner(client.reddit, args.name, options)
        out_prefix = args.output or f"reddit_user_{args.name}"
    else:
        miner = SubredditMiner(client.reddit, args.name, options)
        out_prefix = args.output or f"reddit_sub_{args.name}"

    logging.info(f"Read-only? {client.reddit.read_only}")
    xlsx_path = save_excel(miner.records(), out_prefix)
    return 0 if xlsx_path else 1


if __name__ == "__main__":
    sys.exit(main())
