"""
utils.py — Вспомогательные функции
"""

import logging
import re
import os
import sys
import pandas as pd
from pathlib import Path


# ── Логирование ───────────────────────────────────────────────────────────────

def setup_logging(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """Настраивает и возвращает именованный логгер."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Консоль
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # Файл
        fh = logging.FileHandler(f"logs/{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Текстовая очистка ─────────────────────────────────────────────────────────

_WHITESPACE = re.compile(r"\s+")
_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")


def clean_text(text: str) -> str:
    """
    Очищает текст: убирает лишние пробелы, HTML-сущности.
    Сохраняет армянские Unicode-символы (U+0531–U+058F).
    """
    if not text:
        return ""
    text = _HTML_ENTITY.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def is_armenian(text: str) -> bool:
    """Проверяет, содержит ли текст армянские символы."""
    return bool(re.search(r"[\u0531-\u058F\uFB13-\uFB17]", text))


# ── Работа с датасетом ────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет дубликаты по URL и заголовку."""
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    df = df.drop_duplicates(subset=["title"])
    after = len(df)
    if before != after:
        print(f"Удалено дубликатов: {before - after}")
    return df.reset_index(drop=True)


def load_dataset(path: str = "azatutyun_news.csv") -> pd.DataFrame:
    """Загружает CSV датасет с корректной кодировкой."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Базовая валидация
    required = {"title", "text", "label", "url", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {missing}")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["title"] = df["title"].astype(str)
    return df


def print_dataset_stats(df: pd.DataFrame):
    """Выводит статистику датасета."""
    print("\n" + "═" * 50)
    print("📊 СТАТИСТИКА ДАТАСЕТА")
    print("═" * 50)
    print(f"Всего записей:  {len(df)}")
    print(f"Уникальных URL: {df['url'].nunique()}")
    print("\nРаспределение по категориям:")
    vc = df["label"].value_counts()
    for label, count in vc.items():
        pct = count / len(df) * 100
        bar = "█" * (count // 50)
        print(f"  {label:15s}: {count:5d} ({pct:.1f}%) {bar}")
    print("\nДлина текста (символов):")
    stats = df["text"].str.len().describe()
    print(f"  min:  {stats['min']:.0f}")
    print(f"  mean: {stats['mean']:.0f}")
    print(f"  max:  {stats['max']:.0f}")
    if "date" in df.columns and df["date"].notna().any():
        dates = df["date"].dropna()
        print(f"\nПериод: {dates.min()} — {dates.max()}")
    print("═" * 50 + "\n")


# ── Текстовый препроцессинг для ML ───────────────────────────────────────────

def preprocess_for_ml(text: str) -> str:
    """
    Лёгкий препроцессинг для армянского текста перед TF-IDF.
    Не используем стемминг/лемматизацию (библиотеки не поддерживают армянский).
    """
    if not isinstance(text, str):
        return ""
    # Нижний регистр
    text = text.lower()
    # Убираем URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Убираем числа (опционально — можно оставить)
    text = re.sub(r"\d+", " ", text)
    # Убираем пунктуацию (кроме армянских знаков препинания)
    text = re.sub(r"[^\u0531-\u058F\uFB13-\uFB17\s]", " ", text)
    # Нормализуем пробелы
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def combine_title_text(df: pd.DataFrame, title_weight: int = 3) -> pd.Series:
    """
    Объединяет заголовок и текст. Заголовок повторяется для усиления веса.
    """
    return (df["title"] + " ") * title_weight + df["text"]
