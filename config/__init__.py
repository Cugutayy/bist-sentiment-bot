"""Ortak konfig + logger setup."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"

# .env dosyasını yükle
load_dotenv(ROOT / ".env")


def load_settings() -> dict:
    """settings.yaml'ı parse edip dict döner."""
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


SETTINGS = load_settings()


def setup_logger() -> None:
    """Tüm modüllerde tek tip logger."""
    log_cfg = SETTINGS.get("logging", {})
    log_file = ROOT / log_cfg.get("file", "logs/bist_bot.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_cfg.get("level", "INFO"),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{module}</cyan> | {message}",
    )
    logger.add(
        log_file,
        level=log_cfg.get("level", "INFO"),
        rotation=log_cfg.get("rotation", "1 day"),
        retention=log_cfg.get("retention", "30 days"),
        encoding="utf-8",
    )


setup_logger()


def env(key: str, default: str | None = None) -> str | None:
    """os.environ proxy — yokluğunda default."""
    return os.environ.get(key, default)
