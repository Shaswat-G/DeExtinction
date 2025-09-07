# DeExtinction
Data Mining Scripts for DeExtinction Companies


Auth/config: Loads Reddit credentials from environment.env via dotenv; validates required vars; constructs praw.Reddit client; logs read-only status.
CLI: Modes user or subreddit; args for name, limit, include/exclude comments, polite sleep, output path, env file path, and log level.
Mining (user): Fetches recent submissions and optionally comments via PRAW (submissions.new, comments.new); captures ids, timestamps, subreddit, titles, bodies, scores, num_comments, permalinks; periodic progress + ETA logging; optional inter-item sleep.
Mining (subreddit): Fetches recent posts (subreddit.new) and optionally latest comments (subreddit.comments); same captured fields; progress + ETA logging; optional sleep.
Processing: Builds a consolidated DataFrame; ensures expected columns; converts epoch to UTC and Europe/Zurich timestamps (tz-naive for Excel); sorts by time.
Derivations: Computes engagement_score = score + num_comments for posts; extracts canonical post_id for comments from parent_id/link_id (t3_ parsing).
Join: Produces a Comments+Posts view by joining comments to posts on post_id, adding post_title, post_permalink, post_subreddit, post_engagement_score.
Output: Writes an Excel workbook with three sheets (Posts, Comments, Comments+Posts) using openpyxl; creates output dirs; configurable output filename with sensible defaults (reddit_data/u-<user>.xlsx or r-<sub>.xlsx); logs record counts.
Robustness/UX: Handles missing fields gracefully; time conversion fallback on errors; concise logging with INFO/DEBUG control; typed dataclasses and options with sensible defaults.
