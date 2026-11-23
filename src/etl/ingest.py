import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from pydantic import ValidationError

from src.etl.load import PROCESSED_DATA_PATH, add_discovery_metrics, save_to_parquet
from src.etl.schemas import BookRecord
from src.etl.transform import ALLOWED_LANGUAGES, _normalize_text_key, standardize_genres

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GOOGLE_BOOKS_API_URL = "https://www.googleapis.com/books/v1/volumes"
OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
OPENLIBRARY_WORK_URL = "https://openlibrary.org{work_key}.json"

LANGUAGE_CODE_MAP = {
    "en": "English",
    "eng": "English",
    "pt": "Portuguese",
    "por": "Portuguese",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
}


def _strip_html(text: str) -> str:
    """Remove occasional HTML tags returned in API descriptions."""
    return re.sub(r"<[^>]+>", " ", text)


def fetch_google_books(
    query: str,
    max_results: int = 3,
    api_key: Optional[str] = None,
    max_retries: int = 2,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    """Query Google Books API (fails fast on 429 so fallback can take over)."""
    params: dict[str, Any] = {
        "q": query,
        "maxResults": min(max(1, max_results), 40),
        "printType": "books",
    }
    resolved_key = api_key or os.getenv("GOOGLE_BOOKS_API_KEY")
    if resolved_key:
        params["key"] = resolved_key

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(GOOGLE_BOOKS_API_URL, params=params, timeout=timeout)
            if response.status_code == 429:
                logger.warning("Google Books API rate-limited (429) for query='%s'. Triggering fallback.", query)
                return []
            response.raise_for_status()
            items = response.json().get("items", [])
            logger.info("Google Books API query='%s' returned %d items.", query, len(items))
            return items
        except requests.RequestException as exc:
            if attempt == max_retries:
                logger.warning("Google Books API failed for query='%s': %s", query, exc)
                return []
            time.sleep(2 ** (attempt - 1))
    return []


def parse_google_book_item(item: dict[str, Any]) -> Optional[BookRecord]:
    """Adapt a Google Books volume item into our unified BookRecord contract."""
    volume_id = item.get("id")
    info = item.get("volumeInfo", {})
    if not volume_id or not info:
        return None

    mapped_lang = LANGUAGE_CODE_MAP.get(str(info.get("language", "")).strip().lower())
    if mapped_lang not in ALLOWED_LANGUAGES:
        return None

    authors = info.get("authors") or []
    author_str = ", ".join(a.strip() for a in authors if a and a.strip())
    clean_description = _strip_html(info.get("description") or "")

    image_links = info.get("imageLinks") or {}
    cover_url = image_links.get("thumbnail") or image_links.get("smallThumbnail")
    if isinstance(cover_url, str) and cover_url.startswith("http://"):
        cover_url = cover_url.replace("http://", "https://", 1)

    genres: list[str] = []
    for cat in info.get("categories") or []:
        genres.extend([part.strip() for part in str(cat).split("/") if part.strip()])

    try:
        return BookRecord(
            book_id=f"gb_{volume_id}",
            title=info.get("title", ""),
            author=author_str,
            rating=info.get("averageRating"),
            num_ratings=int(info.get("ratingsCount") or 0),
            ratings_by_stars=[],
            description=clean_description,
            genres=genres,
            pages=info.get("pageCount"),
            language=mapped_lang,
            cover_img=cover_url,
            source="google_books_api",
        )
    except ValidationError:
        return None


def _fetch_openlibrary_description(work_key: str, timeout: int = 10) -> str:
    """Fetch the full book description from an OpenLibrary Work endpoint."""
    try:
        resp = requests.get(OPENLIBRARY_WORK_URL.format(work_key=work_key), timeout=timeout)
        resp.raise_for_status()
        desc = resp.json().get("description", "")
        if isinstance(desc, dict):
            return str(desc.get("value", ""))
        return str(desc or "")
    except requests.RequestException:
        return ""


def fetch_openlibrary_books(
    title: str,
    author: Optional[str] = None,
    max_results: int = 3,
    timeout: int = 10,
) -> list[BookRecord]:
    """Fallback ingestion from OpenLibrary API mapped to our unified BookRecord contract."""
    params: dict[str, Any] = {
        "title": title,
        "limit": max_results,
        "fields": "key,title,author_name,language,number_of_pages_median,ratings_average,ratings_count,subject,cover_i",
    }
    if author:
        params["author"] = author

    try:
        resp = requests.get(OPENLIBRARY_SEARCH_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
    except requests.RequestException as exc:
        logger.error("OpenLibrary API failed for title='%s': %s", title, exc)
        return []

    records: list[BookRecord] = []
    for doc in docs:
        work_key = doc.get("key")
        if not work_key:
            continue

        raw_langs = doc.get("language") or ["eng"]
        mapped_lang = next(
            (LANGUAGE_CODE_MAP[l.lower()] for l in raw_langs if l.lower() in LANGUAGE_CODE_MAP),
            "English",
        )
        if mapped_lang not in ALLOWED_LANGUAGES:
            continue

        description = _strip_html(_fetch_openlibrary_description(work_key, timeout=timeout))
        if len(description.strip()) < 10:
            # Fallback to subject summary if OpenLibrary work has no synopsis yet
            subjects = (doc.get("subject") or [])[:5]
            if subjects:
                description = f"Themes and subjects: {', '.join(subjects)}."

        cover_id = doc.get("cover_i")
        cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None

        try:
            record = BookRecord(
                book_id=f"ol_{work_key.split('/')[-1]}",
                title=doc.get("title", ""),
                author=", ".join(doc.get("author_name") or []),
                rating=doc.get("ratings_average"),
                num_ratings=int(doc.get("ratings_count") or 0),
                ratings_by_stars=[],
                description=description,
                genres=(doc.get("subject") or [])[:6],
                pages=doc.get("number_of_pages_median"),
                language=mapped_lang,
                cover_img=cover_url,
                source="openlibrary_api",
            )
            records.append(record)
        except ValidationError:
            continue

    logger.info("OpenLibrary fallback for title='%s' parsed %d valid BookRecord items.", title, len(records))
    return records


def upsert_books_into_catalog(
    new_records: list[BookRecord],
    catalog_path: Path = PROCESSED_DATA_PATH,
) -> tuple[pd.DataFrame, int]:
    """Idempotently merge new API books into the Parquet catalog without duplicates."""
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found at {catalog_path}. Run `python3 -m src.etl.load` first."
        )

    catalog_df = pd.read_parquet(catalog_path)
    if not new_records:
        logger.info("No valid new records to upsert.")
        return catalog_df, 0

    existing_ids = set(catalog_df["book_id"].astype(str))
    existing_keys = set(
        zip(
            catalog_df["title"].apply(lambda x: _normalize_text_key(x, strip_parentheses=True)),
            catalog_df["author"].apply(_normalize_text_key),
        )
    )

    truly_new_rows: list[dict[str, Any]] = []
    skipped_duplicates = 0

    for rec in new_records:
        norm_key = (
            _normalize_text_key(rec.title, strip_parentheses=True),
            _normalize_text_key(rec.author),
        )
        if rec.book_id in existing_ids or norm_key in existing_keys:
            skipped_duplicates += 1
            continue

        existing_ids.add(rec.book_id)
        existing_keys.add(norm_key)
        truly_new_rows.append(rec.model_dump())

    if not truly_new_rows:
        logger.info(
            "Upsert complete: 0 new books added (%d duplicates skipped).",
            skipped_duplicates,
        )
        return catalog_df, 0

    new_df = standardize_genres(pd.DataFrame(truly_new_rows))

    metric_cols = ["bayesian_rating", "hidden_gem_index"]
    base_catalog = catalog_df.drop(columns=[c for c in metric_cols if c in catalog_df.columns])
    merged_df = pd.concat([base_catalog, new_df], ignore_index=True)

    enriched_df = add_discovery_metrics(merged_df)
    save_to_parquet(enriched_df, catalog_path)

    logger.info(
        "Upsert complete: +%d new books added, %d duplicates skipped (total catalog: %d).",
        len(truly_new_rows),
        skipped_duplicates,
        len(enriched_df),
    )
    return enriched_df, len(truly_new_rows)


def ingest_books(
    queries: list[tuple[str, str]],
    max_results_per_query: int = 2,
    catalog_path: Path = PROCESSED_DATA_PATH,
) -> tuple[pd.DataFrame, int]:
    """Fetch books (Google Books primary, OpenLibrary fallback) and upsert idempotently."""
    logger.info("=== Starting Incremental Ingestion (Google Books + OpenLibrary Fallback) ===")
    parsed_records: list[BookRecord] = []

    for title, author in queries:
        gb_query = f"intitle:{title} inauthor:{author}"
        raw_items = fetch_google_books(query=gb_query, max_results=max_results_per_query)
        gb_records = [r for item in raw_items if (r := parse_google_book_item(item)) is not None]

        if gb_records:
            parsed_records.extend(gb_records)
        else:
            ol_records = fetch_openlibrary_books(
                title=title, author=author, max_results=max_results_per_query
            )
            parsed_records.extend(ol_records)

    updated_df, added_count = upsert_books_into_catalog(parsed_records, catalog_path)
    logger.info("=== Incremental Ingestion Finished ===")
    return updated_df, added_count


if __name__ == "__main__":
    # Smoke test: 1 classic already in Kaggle (should be skipped) + 2 recent releases
    test_queries = [
        ("The Hunger Games", "Suzanne Collins"),
        ("Fourth Wing", "Rebecca Yarros"),
        ("Yellowface", "R. F. Kuang"),
    ]
    df_updated, added = ingest_books(test_queries, max_results_per_query=2)
    print(f"\nAdded {added} new books. Total catalog size: {len(df_updated):,}")

    api_books = df_updated[df_updated["source"] != "kaggle_bbe"]
    if not api_books.empty:
        print("\nSample of API-ingested books in catalog:")
        print(
            api_books[["book_id", "title", "author", "language", "rating", "bayesian_rating", "source"]]
            .tail(10)
            .to_string(index=False)
        )
