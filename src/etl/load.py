import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.etl.transform import RAW_DATA_PATH, run_transform_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = Path("data/processed/books_catalog.parquet")
MIN_VOTES_HIDDEN_GEM = 200
POPULARITY_PENALTY_LAMBDA = 0.15


def add_discovery_metrics(
    df: pd.DataFrame,
    min_votes_hidden_gem: int = MIN_VOTES_HIDDEN_GEM,
    popularity_lambda: float = POPULARITY_PENALTY_LAMBDA,
) -> pd.DataFrame:
    """Compute Bayesian Average (WR) and calibrated Hidden Gem Index for long-tail discovery."""
    df = df.copy()

    catalog_mean = float(df["rating"].mean())
    min_votes_m = float(df["num_ratings"].quantile(0.25))

    v = df["num_ratings"].astype(float)
    r = df["rating"].astype(float)

    # Bayesian Average: WR = (v / (v + m)) * R + (m / (v + m)) * C
    df["bayesian_rating"] = (
        (v / (v + min_votes_m)) * r + (min_votes_m / (v + min_votes_m)) * catalog_mean
    ).round(4)

    # Calibrated Hidden Gem Index: WR - lambda * log10(v + 1)
    df["hidden_gem_index"] = np.where(
        df["num_ratings"] >= min_votes_hidden_gem,
        (df["bayesian_rating"] - popularity_lambda * np.log10(v + 1.0)).round(4),
        np.nan,
    )

    eligible_count = int(df["hidden_gem_index"].notna().sum())
    logger.info(
        "Discovery metrics computed | C=%.4f | m=%.1f (p25) | lambda=%.2f | eligible hidden gems (v>=%d): %d",
        catalog_mean,
        min_votes_m,
        popularity_lambda,
        min_votes_hidden_gem,
        eligible_count,
    )
    return df


def save_to_parquet(
    df: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_PATH,
) -> Path:
    """Idempotently save the enriched catalog to Parquet with clean dtypes."""
    df = df.copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "pages" in df.columns:
        df["pages"] = df["pages"].astype("Int64")

    df = df.drop_duplicates(subset=["book_id"], keep="first").reset_index(drop=True)

    tmp_path = output_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False, engine="pyarrow")
    tmp_path.replace(output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Saved %d books idempotently to %s (%.2f MB).",
        len(df),
        output_path,
        file_size_mb,
    )
    return output_path


def run_etl_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Run the complete Extract -> Transform -> Enrich -> Load pipeline."""
    logger.info("=== Starting Full ETL Pipeline (Week 3) ===")
    df_clean = run_transform_pipeline(raw_path)
    df_enriched = add_discovery_metrics(df_clean)
    save_to_parquet(df_enriched, output_path)
    logger.info("=== Full ETL Pipeline Finished Successfully ===")
    return df_enriched


if __name__ == "__main__":
    catalog_df = run_etl_pipeline()

    print("\n=== TOP 10 BY BAYESIAN RATING (HIGH CONFIDENCE QUALITY) ===")
    print(
        catalog_df.sort_values(by="bayesian_rating", ascending=False)[
            ["title", "author", "rating", "num_ratings", "bayesian_rating"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n=== TOP 10 HIDDEN GEMS (CALIBRATED LONG TAIL) ===")
    print(
        catalog_df.dropna(subset=["hidden_gem_index"])
        .sort_values(by="hidden_gem_index", ascending=False)[
            ["title", "author", "rating", "num_ratings", "bayesian_rating", "hidden_gem_index"]
        ]
        .head(10)
        .to_string(index=False)
    )
