import logging
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.etl.schemas import BookRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/books_1.Best_Books_Ever.csv")
ALLOWED_LANGUAGES = {"English", "Portuguese"}

COLUMN_MAPPING = {
    "bookId": "book_id",
    "title": "title",
    "author": "author",
    "rating": "rating",
    "numRatings": "num_ratings",
    "ratingsByStars": "ratings_by_stars",
    "description": "description",
    "genres": "genres",
    "pages": "pages",
    "language": "language",
    "coverImg": "cover_img",
}


def _clean_nan(value: Any) -> Any:
    """Convert pandas/float NaN values to None for clean Pydantic validation."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _normalize_text_key(text: str, strip_parentheses: bool = False) -> str:
    """Normalize text for entity resolution (lowercasing and whitespace cleanup)."""
    cleaned = str(text).lower()
    if strip_parentheses:
        cleaned = re.sub(r"\s*\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_raw_catalog(raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Goodreads CSV dataset with error handling."""
    logger.info("Loading raw dataset from: %s", raw_path)
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError as exc:
        logger.error("Raw dataset not found at %s", raw_path)
        raise exc
    except Exception as exc:
        logger.error("Failed to read CSV file: %s", exc)
        raise exc

    logger.info("Loaded %d rows and %d columns.", len(df), len(df.columns))
    return df


def filter_languages(
    df: pd.DataFrame, allowed_languages: set[str] = ALLOWED_LANGUAGES
) -> pd.DataFrame:
    """Keep only books in the allowed target languages."""
    initial_count = len(df)
    lang_series = df["language"].fillna("").astype(str).str.strip()
    filtered = df[lang_series.isin(allowed_languages)].copy()
    dropped = initial_count - len(filtered)

    logger.info(
        "Language filter (%s): kept %d rows, dropped %d rows.",
        ", ".join(sorted(allowed_languages)),
        len(filtered),
        dropped,
    )
    return filtered


def validate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean records using the BookRecord Pydantic contract."""
    selected = df[list(COLUMN_MAPPING.keys())].rename(columns=COLUMN_MAPPING)
    valid_records: list[dict[str, Any]] = []
    invalid_count = 0

    for raw_row in selected.to_dict(orient="records"):
        cleaned_row = {k: _clean_nan(v) for k, v in raw_row.items()}
        try:
            record = BookRecord(**cleaned_row)
            valid_records.append(record.model_dump())
        except ValidationError:
            invalid_count += 1

    logger.info(
        "Pydantic validation: %d valid books, %d invalid records dropped.",
        len(valid_records),
        invalid_count,
    )
    return pd.DataFrame(valid_records)


def standardize_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize genre tags and remove duplicates within each book."""
    df = df.copy()

    def _clean_genre_list(genres: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for genre in genres:
            tag = " ".join(genre.split()).title()
            if tag and tag not in seen:
                seen.add(tag)
                normalized.append(tag)
        return normalized

    df["genres"] = df["genres"].apply(_clean_genre_list)
    return df


def deduplicate_books(df: pd.DataFrame) -> pd.DataFrame:
    """Apply 2-stage entity resolution to remove exact and translation duplicates."""
    df = df.copy()
    initial_count = len(df)

    # Prioritize records with more ratings and richer descriptions
    df["_desc_len"] = df["description"].str.len()
    df = df.sort_values(by=["num_ratings", "_desc_len"], ascending=[False, False])

    # Stage 1: Exact normalized (title, author) deduplication
    df["_norm_title"] = df["title"].apply(
        lambda x: _normalize_text_key(x, strip_parentheses=True)
    )
    df["_norm_author"] = df["author"].apply(_normalize_text_key)

    before_stage1 = len(df)
    df = df.drop_duplicates(subset=["_norm_title", "_norm_author"], keep="first")
    stage1_dropped = before_stage1 - len(df)

    # Stage 2: Edition/translation resolution for (author, num_ratings > 100)
    high_vol = df[df["num_ratings"] > 100].drop_duplicates(
        subset=["_norm_author", "num_ratings"], keep="first"
    )
    low_vol = df[df["num_ratings"] <= 100]

    before_stage2 = len(df)
    df = pd.concat([high_vol, low_vol], ignore_index=True)
    stage2_dropped = before_stage2 - len(df)

    # Clean temporary helper columns and restore deterministic ordering
    df = (
        df.drop(columns=["_desc_len", "_norm_title", "_norm_author"])
        .sort_values(by="num_ratings", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "Entity resolution: removed %d normalized title/author duplicates and %d edition/translation duplicates (total: %d -> %d).",
        stage1_dropped,
        stage2_dropped,
        initial_count,
        len(df),
    )
    return df


def run_transform_pipeline(raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Execute the full Extract + Transform pipeline and return the clean DataFrame."""
    logger.info("=== Starting ETL Transform Pipeline ===")
    df_raw = load_raw_catalog(raw_path)
    df_lang = filter_languages(df_raw)
    df_valid = validate_records(df_lang)
    df_genres = standardize_genres(df_valid)
    df_clean = deduplicate_books(df_genres)
    logger.info("=== ETL Transform Pipeline Finished: %d clean books ready ===", len(df_clean))
    return df_clean


if __name__ == "__main__":
    clean_df = run_transform_pipeline()
    print("\nSample of cleaned catalog (top 5 by ratings):")
    print(
        clean_df[["title", "author", "language", "rating", "num_ratings", "pages"]]
        .head(5)
        .to_string(index=False)
    )
