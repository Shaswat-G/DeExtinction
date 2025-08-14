import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import logging
from typing import Tuple, Dict, Any


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_data(input_path: str, expected_cols: list) -> pd.DataFrame:
    """Load CSV data and ensure all expected columns exist."""
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Loaded data from {input_path} with {df.shape[0]} rows.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def process_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert created_utc to datetime columns and add date parts."""
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df["created_dt_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    try:
        df["created_dt_ch"] = df["created_dt_utc"].dt.tz_convert("Europe/Zurich")
    except Exception as e:
        logging.warning(f"Timezone conversion failed: {e}")
        df["created_dt_ch"] = df["created_dt_utc"]
    df["date_ch"] = df["created_dt_ch"].dt.date
    df["hour_ch"] = df["created_dt_ch"].dt.hour
    df["dow_ch"] = df["created_dt_ch"].dt.day_name()
    return df


def extract_post_id(row: pd.Series) -> Any:
    """Derive post_id for posts and comments."""
    if row["type"] == "post":
        return row["id"]
    lid = str(row.get("link_id", ""))
    m = re.match(r"t3_(.+)", lid)
    return m.group(1) if m else np.nan


def build_tidy_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build tidy posts and comments tables."""
    posts = df[df["type"] == "post"].copy()
    comments = df[df["type"] == "comment"].copy()
    posts["content"] = (
        posts["title"].fillna("") + "\n\n" + posts["selftext"].fillna("")
    ).str.strip()
    comments["content"] = comments["body"].fillna("").str.strip()
    posts["word_count"] = posts["content"].str.split().apply(len)
    comments["word_count"] = comments["content"].str.split().apply(len)
    posts["num_comments"] = pd.to_numeric(posts["num_comments"], errors="coerce")
    posts["score"] = pd.to_numeric(posts["score"], errors="coerce")
    posts["engagement"] = posts[["score", "num_comments"]].fillna(0).sum(axis=1)
    comments["score"] = pd.to_numeric(comments["score"], errors="coerce")
    posts_tidy_cols = [
        "post_id",
        "created_dt_utc",
        "created_dt_ch",
        "subreddit",
        "title",
        "selftext",
        "url",
        "permalink",
        "score",
        "num_comments",
        "engagement",
        "word_count",
        "date_ch",
        "hour_ch",
        "dow_ch",
    ]
    comments_tidy_cols = [
        "id",
        "post_id",
        "created_dt_utc",
        "created_dt_ch",
        "subreddit",
        "link_title",
        "content",
        "score",
        "permalink",
        "parent_id",
        "word_count",
        "date_ch",
        "hour_ch",
        "dow_ch",
    ]
    posts_tidy = posts[posts_tidy_cols].sort_values("created_dt_utc")
    comments_tidy = comments[comments_tidy_cols].sort_values("created_dt_utc")
    return posts_tidy, comments_tidy


def join_comments_to_posts(
    comments_tidy: pd.DataFrame, posts_tidy: pd.DataFrame
) -> pd.DataFrame:
    """Join comments to their parent post metadata."""
    return comments_tidy.merge(
        posts_tidy[["post_id", "title", "permalink"]].rename(
            columns={"title": "post_title", "permalink": "post_permalink"}
        ),
        on="post_id",
        how="left",
    )


def analytics(
    df: pd.DataFrame, posts_tidy: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute top subreddits, top posts, and daily activity."""
    sub_counts = (
        df.groupby("subreddit")["id"]
        .count()
        .sort_values(ascending=False)
        .rename("items_count")
        .reset_index()
    )
    top_posts = posts_tidy.sort_values(["engagement", "score"], ascending=False).head(
        10
    )
    daily_counts = (
        df.groupby("date_ch")["id"].count().rename("items_per_day").reset_index()
    )
    return sub_counts, top_posts, daily_counts


def plot_daily_activity(daily_counts: pd.DataFrame, output_dir: str) -> str:
    """Plot daily activity and save to file."""
    plot_path = os.path.join(output_dir, "daily_activity.png")
    if not daily_counts.empty:
        plt.figure()
        plt.plot(daily_counts["date_ch"], daily_counts["items_per_day"])
        plt.title("Daily Activity (items per day)")
        plt.xlabel("Date (Europe/Zurich)")
        plt.ylabel("Posts + Comments")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved plot to {plot_path}")
    else:
        logging.warning("No daily activity data to plot.")
    return plot_path


def save_outputs(
    posts_tidy: pd.DataFrame,
    comments_tidy: pd.DataFrame,
    comments_joined: pd.DataFrame,
    long_df: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    """Save cleaned outputs to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    posts_out = os.path.join(output_dir, "colossal_reddit_clean_posts.csv")
    comments_out = os.path.join(output_dir, "colossal_reddit_clean_comments.csv")
    comments_joined_out = os.path.join(
        output_dir, "colossal_reddit_comments_joined.csv"
    )
    long_out = os.path.join(output_dir, "colossal_reddit_long.csv")
    posts_tidy.to_csv(posts_out, index=False, encoding="utf-8")
    comments_tidy.to_csv(comments_out, index=False, encoding="utf-8")
    comments_joined.to_csv(comments_joined_out, index=False, encoding="utf-8")
    long_df.to_csv(long_out, index=False, encoding="utf-8")
    logging.info(f"Saved outputs to {output_dir}")
    return {
        "posts_csv": posts_out,
        "comments_csv": comments_out,
        "comments_joined_csv": comments_joined_out,
        "long_form_csv": long_out,
    }


def build_long_form(
    posts_tidy: pd.DataFrame, comments_joined: pd.DataFrame
) -> pd.DataFrame:
    """Stack posts and comments into a long-form DataFrame."""
    posts_long = posts_tidy.rename(
        columns={"title": "title_or_link_title", "selftext": "body_or_selftext"}
    )[
        [
            "post_id",
            "created_dt_ch",
            "subreddit",
            "title_or_link_title",
            "body_or_selftext",
            "score",
            "num_comments",
            "engagement",
            "permalink",
        ]
    ]
    posts_long["kind"] = "post"
    comments_long = comments_joined.rename(
        columns={"link_title": "title_or_link_title", "content": "body_or_selftext"}
    )[
        [
            "post_id",
            "created_dt_ch",
            "subreddit",
            "title_or_link_title",
            "body_or_selftext",
            "score",
            "permalink",
        ]
    ]
    comments_long["engagement"] = np.nan
    comments_long["num_comments"] = np.nan
    comments_long["kind"] = "comment"
    long_df = pd.concat([posts_long, comments_long], ignore_index=True).sort_values(
        "created_dt_ch"
    )
    return long_df


def save_analysis_to_excel(
    sub_counts: pd.DataFrame,
    top_posts: pd.DataFrame,
    comments_joined: pd.DataFrame,
    posts_tidy: pd.DataFrame,
    comments_tidy: pd.DataFrame,
    long_df: pd.DataFrame,
    daily_counts: pd.DataFrame,
    output_dir: str,
) -> str:
    """Save all analysis tables to a single Excel workbook with each table on a different sheet."""
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, "colossal_reddit_analysis.xlsx")
    def make_tz_naive(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
        return df

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        make_tz_naive(sub_counts).to_excel(writer, sheet_name="Top Subreddits", index=False)
        make_tz_naive(top_posts).to_excel(writer, sheet_name="Top Posts", index=False)
        make_tz_naive(comments_joined.head(1000)).to_excel(writer, sheet_name="Comments+Posts", index=False)
        make_tz_naive(posts_tidy).to_excel(writer, sheet_name="Posts Tidy", index=False)
        make_tz_naive(comments_tidy).to_excel(writer, sheet_name="Comments Tidy", index=False)
        make_tz_naive(long_df).to_excel(writer, sheet_name="Long Form", index=False)
        make_tz_naive(daily_counts).to_excel(writer, sheet_name="Daily Activity", index=False)
    logging.info(f"Saved all analysis tables to {excel_path}")
    return excel_path


def summarize(
    df: pd.DataFrame,
    posts_tidy: pd.DataFrame,
    comments_tidy: pd.DataFrame,
    outputs: Dict[str, str],
) -> Dict[str, Any]:
    """Return summary statistics."""
    return {
        "rows_total": int(df.shape[0]),
        "posts": int(posts_tidy.shape[0]),
        "comments": int(comments_tidy.shape[0]),
        "subreddits": int(df["subreddit"].nunique()),
        "date_range_ch": f"{df['created_dt_ch'].min()} → {df['created_dt_ch'].max()}",
        "outputs": outputs,
    }


def main(input_path: str, output_dir: str) -> Dict[str, Any]:
    """Main processing function for Reddit CSV analytics."""
    setup_logging()
    expected_cols = [
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
        "link_title",
        "body",
        "parent_id",
    ]
    df = load_data(input_path, expected_cols)
    df = process_time(df)
    df["post_id"] = df.apply(extract_post_id, axis=1)
    posts_tidy, comments_tidy = build_tidy_tables(df)
    comments_joined = join_comments_to_posts(comments_tidy, posts_tidy)
    sub_counts, top_posts, daily_counts = analytics(df, posts_tidy)
    plot_path = plot_daily_activity(daily_counts, output_dir)
    long_df = build_long_form(posts_tidy, comments_joined)
    outputs = save_outputs(
        posts_tidy, comments_tidy, comments_joined, long_df, output_dir
    )
    outputs["daily_activity_plot"] = plot_path
    excel_path = save_analysis_to_excel(
        sub_counts,
        top_posts,
        comments_joined,
        posts_tidy,
        comments_tidy,
        long_df,
        daily_counts,
        output_dir,
    )
    outputs["excel_analysis"] = excel_path
    summary = summarize(df, posts_tidy, comments_tidy, outputs)
    logging.info(f"Summary: {summary}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and analyze Reddit CSV data.")
    parser.add_argument(
        "--input", type=str, default="colossal_reddit.csv", help="Input CSV file path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save outputs"
    )
    args = parser.parse_args()
    main(args.input, args.output_dir)
