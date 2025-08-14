import os, sys
from dotenv import load_dotenv
import pandas as pd
import praw

load_dotenv("environment.env")
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
    sys.exit("Missing env vars. Check .env.")

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)

print("Read-only? ", reddit.read_only)  # should be True

u = reddit.redditor("ColossalBiosciences")

def gen_posts(user):
    # Auto-paginates under the hood, subject to rate limits.
    for s in user.submissions.new(limit=None):
        yield {
            "type":"post",
            "id": s.id,
            "created_utc": s.created_utc,
            "subreddit": s.subreddit.display_name,
            "title": s.title,
            "selftext": s.selftext,
            "url": s.url,
            "score": s.score,
            "num_comments": s.num_comments,
            "permalink": "https://www.reddit.com" + s.permalink
        }

def gen_comments(user):
    for c in user.comments.new(limit=None):
        yield {
            "type":"comment",
            "id": c.id,
            "created_utc": c.created_utc,
            "subreddit": c.subreddit.display_name,
            "link_id": c.link_id,
            "link_title": getattr(c.submission, "title", None),
            "body": c.body,
            "score": c.score,
            "parent_id": c.parent_id,
            "permalink": "https://www.reddit.com" + c.permalink
        }

rows = list(gen_posts(u)) + list(gen_comments(u))
df = pd.DataFrame(rows)

# Sort, save compact
df = df.sort_values("created_utc")
df.to_csv("colossal_reddit.csv", index=False, encoding="utf-8")
df.to_parquet("colossal_reddit.parquet", index=False)

print(f"Saved {len(df)} rows → colossal_reddit.csv & colossal_reddit.parquet")
