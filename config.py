from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Path to your DuckDB file — override via env var DB_PATH or .env
    db_path: str = str(Path(__file__).parent / "db/rwitc.db")

    # API metadata
    api_title: str = "RWITC Horse Racing API"
    api_version: str = "0.1.0"
    api_description: str = (
        "REST API for the RWITC horse-racing database. "
        "Provides read access to races, runners, horses, jockeys, trainers, venues, "
        "meetings, ratings, dividends, and regulatory data."
    )

    # Pagination defaults
    default_limit: int = 50
    max_limit: int = 500

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
