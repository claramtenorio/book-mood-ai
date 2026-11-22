import ast
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class BookRecord(BaseModel):
    """Validated data contract for a single book record in our pipeline."""

    book_id: str = Field(..., min_length=1, description="Unique identifier from source")
    title: str = Field(..., min_length=1)
    author: str = Field(..., min_length=1)
    rating: float = Field(..., gt=0.0, le=5.0)
    num_ratings: int = Field(..., gt=0)
    ratings_by_stars: list[int] = Field(default_factory=list)
    description: str = Field(..., min_length=10)
    genres: list[str] = Field(default_factory=list)
    pages: Optional[int] = Field(default=None, gt=0)
    language: str = Field(..., min_length=2)
    cover_img: Optional[str] = None
    source: str = Field(default="kaggle_bbe")

    @field_validator("genres", mode="before")
    @classmethod
    def parse_genres(cls, value: object) -> list[str]:
        """Convert stringified lists like \"['Fiction', 'Fantasy']\" into Python lists."""
        if isinstance(value, list):
            return [str(g).strip() for g in value if str(g).strip()]
        if isinstance(value, str) and value.strip():
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(g).strip() for g in parsed if str(g).strip()]
            except (ValueError, SyntaxError):
                return []
        return []

    @field_validator("ratings_by_stars", mode="before")
    @classmethod
    def parse_ratings_by_stars(cls, value: object) -> list[int]:
        """Convert stringified star counts like \"['3444695', ...]\" into list of ints."""
        if isinstance(value, list):
            return [int(x) for x in value if str(x).isdigit()]
        if isinstance(value, str) and value.strip():
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [int(x) for x in parsed if str(x).isdigit()]
            except (ValueError, SyntaxError):
                return []
        return []

    @field_validator("pages", mode="before")
    @classmethod
    def parse_pages(cls, value: object) -> Optional[int]:
        """Handle messy page values (e.g. '374', '374.0', '1 page', NaN)."""
        if value is None or str(value).strip().lower() in {"", "nan", "none"}:
            return None
        try:
            num = int(float(str(value).split()[0]))
            return num if num > 0 else None
        except (ValueError, IndexError):
            return None

    @field_validator("title", "author", "description", "language", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> str:
        """Normalize whitespace on required text fields."""
        if value is None or str(value).strip().lower() == "nan":
            return ""
        return " ".join(str(value).split())
