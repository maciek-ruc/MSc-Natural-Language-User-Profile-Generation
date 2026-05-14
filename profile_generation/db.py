from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mysql import connector

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()


def get_sql_config() -> dict[str, Any]:
    load_env()
    return {
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME"),
    }


def get_connection() -> Any:
    return connector.connect(**get_sql_config())