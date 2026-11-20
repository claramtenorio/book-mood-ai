from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/books_1.Best_Books_Ever.csv")


def run_eda() -> None:
    """Run initial exploratory data analysis on the raw Goodreads dataset."""
    df = pd.read_csv(RAW_PATH)

    print("=== 1. DATASET DIMENSIONS ===")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print("\nAvailable columns:")
    print(list(df.columns))

    print("\n=== 2. MISSING VALUES PERCENTAGE (TOP 10) ===")
    null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    print(null_pct.head(10).round(2))

    print("\n=== 3. SUMMARY STATISTICS (RATING & VOLUME) ===")
    print(df[["rating", "numRatings"]].describe().round(2))

    print("\n=== 4. SAMPLE FORMAT (GENRES & RATINGS BY STARS) ===")
    print(df[["title", "author", "genres", "ratingsByStars"]].head(3).to_string())


if __name__ == "__main__":
    run_eda()
