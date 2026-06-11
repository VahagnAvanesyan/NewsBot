"""
scraper.py — Сборщик новостей с azatutyun.am
Категории: Սպորտ (Sport, z/1517) и Քաղաքական (Politics, z/1516)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import os
import random
from datetime import datetime
from utils import setup_logging, clean_text, remove_duplicates

logger = setup_logging("scraper")

# ── Конфиг ────────────────────────────────────────────────────────────────────

BASE_URL = "https://www.azatutyun.am"

CATEGORIES = {
    "Սպորտ":       "/z/15473",   # Sport section ID
    "Քաղաքական":   "/z/2039",   # Politics section ID
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "hy,ru;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.azatutyun.am/",
    "DNT": "1",
}

DELAY_MIN = 1.5   # секунд между запросами
DELAY_MAX = 3.5
MAX_RETRIES = 3
TIMEOUT = 20


# ── Сессия ────────────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


# ── Получение страницы ────────────────────────────────────────────────────────

def fetch_page(session: requests.Session, url: str) -> BeautifulSoup | None:
    """Загружает страницу с повторными попытками. Возвращает BeautifulSoup или None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return BeautifulSoup(resp.text, "html.parser")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP {e.response.status_code} для {url} (попытка {attempt})")
            if e.response.status_code in (403, 404):
                return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ошибка запроса {url}: {e} (попытка {attempt})")
        time.sleep(DELAY_MIN * attempt)
    logger.error(f"Не удалось загрузить: {url}")
    return None


def polite_sleep():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


# ── Парсинг списка статей ─────────────────────────────────────────────────────

def parse_article_links(soup: BeautifulSoup) -> list[str]:
    """
    Извлекает ссылки на статьи из страницы-раздела.
    azatutyun.am использует несколько шаблонов разметки.
    """
    urls = []
    selectors = [
        "ul.media-block-list li a",
        "div.content-offset a",
        "article a",
        "h4.media-block-title a",
        "h3.title a",
        "a.img-wrap",
    ]
    for sel in selectors:
        for tag in soup.select(sel):
            href = tag.get("href", "")
            # Статьи имеют формат /a/NNNNNNN.html
            if href and "/a/" in href:
                full = href if href.startswith("http") else BASE_URL + href
                urls.append(full)

    # Дедупликация с сохранением порядка
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def get_next_page_url(soup: BeautifulSoup, section_path: str, current_page: int) -> str | None:
    """
    azatutyun.am использует параметр ?p=N для пагинации раздела.
    """
    # Поиск кнопки «следующая страница»
    next_btn = soup.find("a", class_="next") or soup.find("a", string=lambda t: t and ">" in t)
    if next_btn and next_btn.get("href"):
        href = next_btn["href"]
        return href if href.startswith("http") else BASE_URL + href

    # Fallback: инкрементируем параметр p
    next_page = current_page + 1
    if next_page <= 500:   # разумный лимит
        return f"{BASE_URL}{section_path}?p={next_page}"
    return None


# ── Парсинг отдельной статьи ──────────────────────────────────────────────────

def parse_article(soup: BeautifulSoup, url: str, label: str) -> dict | None:
    """Извлекает заголовок, текст, дату из страницы статьи."""
    try:
        # Заголовок
        title_tag = (
            soup.find("h1", class_="title")
            or soup.find("h1")
            or soup.find("h2", class_="title")
        )
        title = clean_text(title_tag.get_text()) if title_tag else ""
        if not title:
            logger.debug(f"Нет заголовка: {url}")
            return None

        # Дата публикации
        date_tag = (
            soup.find("time")
            or soup.find(class_="published")
            or soup.find(class_="datetime")
            or soup.find("span", class_="date")
        )
        if date_tag:
            date_str = date_tag.get("datetime") or clean_text(date_tag.get_text())
        else:
            date_str = ""

        # Нормализуем дату
        date = parse_date(date_str)

        # Основной текст статьи
        body = (
            soup.find("div", class_="body-text")
            or soup.find("div", class_="article-content")
            or soup.find("div", class_="wsw")
            or soup.find("article")
            or soup.find("div", id="article-content")
        )
        if body:
            # Удаляем скрипты, рекламу, навигацию
            for tag in body.find_all(["script", "style", "aside", "nav", "figure"]):
                tag.decompose()
            text = clean_text(body.get_text(separator=" "))
        else:
            text = ""

        if len(text) < 50:
            logger.debug(f"Слишком короткий текст ({len(text)} симв.): {url}")
            return None

        return {
            "title": title,
            "text":  text,
            "label": label,
            "url":   url,
            "date":  date,
        }

    except Exception as e:
        logger.error(f"Ошибка парсинга статьи {url}: {e}")
        return None


def parse_date(raw: str) -> str:
    """Пробует распознать дату из строки."""
    if not raw:
        return ""
    raw = raw.strip()
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d %B %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw[:19], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw[:20]   # возвращаем как есть, обрезав


# ── Основной цикл скрапинга ───────────────────────────────────────────────────

def scrape_category(
    session: requests.Session,
    label: str,
    section_path: str,
    target: int = 5000,
    checkpoint_file: str = "",
) -> list[dict]:
    """
    Собирает статьи одной категории.
    Поддерживает checkpoint: если файл существует — продолжает с него.
    """
    articles = []
    seen_urls: set[str] = set()

    # Загружаем checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        df_cp = pd.read_csv(checkpoint_file, encoding="utf-8")
        articles = df_cp.to_dict("records")
        seen_urls = {a["url"] for a in articles}
        logger.info(f"[{label}] Checkpoint загружен: {len(articles)} статей")

    page = 1
    consecutive_empty = 0

    while len(articles) < target:
        page_url = f"{BASE_URL}{section_path}?p={page}" if page > 1 else f"{BASE_URL}{section_path}"
        logger.info(f"[{label}] Страница {page}: {page_url} ({len(articles)}/{target})")

        soup = fetch_page(session, page_url)
        if soup is None:
            consecutive_empty += 1
            if consecutive_empty >= 5:
                logger.warning(f"[{label}] 5 пустых страниц подряд — завершаю")
                break
            page += 1
            polite_sleep()
            continue

        links = parse_article_links(soup)
        if not links:
            consecutive_empty += 1
            logger.warning(f"[{label}] Нет ссылок на стр. {page} (пустых подряд: {consecutive_empty})")
            if consecutive_empty >= 5:
                break
            page += 1
            polite_sleep()
            continue

        consecutive_empty = 0
        new_on_page = 0

        for art_url in links:
            if art_url in seen_urls:
                continue
            seen_urls.add(art_url)

            art_soup = fetch_page(session, art_url)
            if art_soup is None:
                polite_sleep()
                continue

            article = parse_article(art_soup, art_url, label)
            if article:
                articles.append(article)
                new_on_page += 1
                logger.debug(f"  ✓ {article['title'][:60]}")

            polite_sleep()

            # Промежуточное сохранение каждые 100 статей
            if checkpoint_file and len(articles) % 100 == 0 and len(articles) > 0:
                pd.DataFrame(articles).to_csv(checkpoint_file, index=False, encoding="utf-8")
                logger.info(f"[{label}] Checkpoint сохранён: {len(articles)} статей")

        logger.info(f"[{label}] Стр. {page}: +{new_on_page} | итого {len(articles)}")
        page += 1
        polite_sleep()

    # Финальное сохранение checkpoint
    if checkpoint_file and articles:
        pd.DataFrame(articles).to_csv(checkpoint_file, index=False, encoding="utf-8")

    return articles


# ── Главная функция ───────────────────────────────────────────────────────────

def run_scraper(target_per_class: int = 5000, output_file: str = "azatutyun_news.csv"):
    """
    Запускает полный цикл сбора данных для обеих категорий.
    """
    session = make_session()
    all_articles = []

    for label, section_path in CATEGORIES.items():
        checkpoint = f"checkpoint_{label}.csv"
        logger.info(f"═══ Начинаю сбор: {label} (цель: {target_per_class}) ═══")

        articles = scrape_category(
            session=session,
            label=label,
            section_path=section_path,
            target=target_per_class,
            checkpoint_file=checkpoint,
        )
        logger.info(f"[{label}] Собрано: {len(articles)} статей")
        all_articles.extend(articles)

    if not all_articles:
        logger.error("Не удалось собрать ни одной статьи!")
        return

    df = pd.DataFrame(all_articles)
    df = remove_duplicates(df)
    df = df[["title", "text", "label", "url", "date"]]
    df.to_csv(output_file, index=False, encoding="utf-8-sig")   # utf-8-sig для корректного открытия в Excel

    logger.info(f"═══ Готово! Сохранено {len(df)} статей в {output_file} ═══")
    logger.info(f"Распределение:\n{df['label'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scraper azatutyun.am")
    parser.add_argument("--target", type=int, default=5000, help="Статей на категорию")
    parser.add_argument("--output", type=str, default="azatutyun_news.csv", help="Выходной CSV файл")
    args = parser.parse_args()
    run_scraper(target_per_class=args.target, output_file=args.output)
