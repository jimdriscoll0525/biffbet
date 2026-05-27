"""Shared helpers: paths, config loading, env, and structured logging.

This module is import-light on purpose so every other module can depend on it
without pulling in heavy data libraries.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.logging import RichHandler

# --- Paths -------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent
CONFIG_PATH: Path = PACKAGE_DIR / "config.yaml"
STORAGE_DIR: Path = PACKAGE_DIR / "storage"
CACHE_DIR: Path = STORAGE_DIR / "cache"
DB_PATH: Path = STORAGE_DIR / "mlb_value_bot.db"

# Load .env once, from the package directory or its parent (repo root).
load_dotenv(PACKAGE_DIR / ".env")
load_dotenv(PACKAGE_DIR.parent / ".env")


def ensure_dirs() -> None:
    """Create storage/cache directories if they don't exist."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load and cache config.yaml as a plain dict.

    Cached so repeated calls within a run are free; pass an explicit path to
    bypass the default (used by tests).
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable (already populated from .env)."""
    return os.getenv(name, default)


def get_bankroll(default: float = 1000.0) -> float:
    """Bankroll from the BANKROLL env var, used for Kelly stake sizing."""
    raw = os.getenv("BANKROLL")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_LOGGING_CONFIGURED = False


def setup_logging(config: dict[str, Any] | None = None) -> logging.Logger:
    """Configure root logging once: pretty console (rich) + rotating-ish file.

    Idempotent — safe to call from every CLI command.
    """
    global _LOGGING_CONFIGURED
    logger = logging.getLogger("mlb_value_bot")
    if _LOGGING_CONFIGURED:
        return logger

    config = config or load_config()
    log_cfg = config.get("logging", {})
    level_name = os.getenv("LOG_LEVEL") or log_cfg.get("level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)

    ensure_dirs()
    log_file = PACKAGE_DIR / log_cfg.get("file", "storage/mlb_value_bot.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(level)
    logger.handlers.clear()

    # Console: rich handler (clean, colorized).
    console_handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File: plain, parseable, append.
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    logger.propagate = False
    _LOGGING_CONFIGURED = True
    return logger


def get_logger(name: str = "mlb_value_bot") -> logging.Logger:
    """Return a child logger; ensures logging is configured first."""
    setup_logging()
    return logging.getLogger(name if name.startswith("mlb_value_bot") else f"mlb_value_bot.{name}")
