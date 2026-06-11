"""
bot.py — Telegram-бот для классификации армянских новостей
Библиотека: python-telegram-bot v20+ (async)

Команды:
  /start  — приветствие
  /help   — справка
  /stats  — статистика модели
  Текст   — классифицировать новость
  URL     — скачать и классифицировать статью по ссылке
"""

import os
import sys
import asyncio
import logging
import pickle
import re
from pathlib import Path

# python-telegram-bot v20+
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode

from utils import setup_logging, preprocess_for_ml, clean_text
from train import load_model, predict_text, MODEL_DIR

logger = setup_logging("bot")

# ── Конфиг ────────────────────────────────────────────────────────────────────

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
USE_XGBOOST = os.getenv("USE_XGBOOST", "1") == "1"

# Эмодзи для категорий
CATEGORY_EMOJI = {
    "Սպորտ":      "⚽",
    "Քաղաքական":  "🏛️",
}

# Минимальная длина текста для классификации
MIN_TEXT_LEN = 20


# ── Загрузка модели ───────────────────────────────────────────────────────────

def load_model_safe():
    """Пробует XGBoost, при неудаче — Baseline."""
    if USE_XGBOOST and (MODEL_DIR / "xgboost_pipeline.pkl").exists():
        try:
            pipeline, le = load_model(use_xgboost=True)
            logger.info("Модель загружена: XGBoost")
            return pipeline, le, "XGBoost"
        except Exception as e:
            logger.warning(f"Не удалось загрузить XGBoost: {e}")

    pipeline, le = load_model(use_xgboost=False)
    logger.info("Модель загружена: Baseline (TF-IDF + LR)")
    return pipeline, le, "Baseline"


# ── Утилиты ───────────────────────────────────────────────────────────────────

def is_url(text: str) -> bool:
    return bool(re.match(r"https?://", text.strip()))


def fetch_article_text(url: str) -> str:
    """Скачивает текст статьи по URL."""
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url.strip(), headers=headers, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # Убираем мусор
        for tag in soup.find_all(["script", "style", "nav", "aside", "footer"]):
            tag.decompose()

        # Заголовок
        h1 = soup.find("h1")
        title = clean_text(h1.get_text()) if h1 else ""

        # Тело статьи
        body = (
            soup.find("div", class_="body-text")
            or soup.find("div", class_="article-content")
            or soup.find("div", class_="wsw")
            or soup.find("article")
        )
        body_text = clean_text(body.get_text(separator=" ")) if body else ""

        return (title + " " + body_text).strip()
    except Exception as e:
        logger.error(f"Ошибка при загрузке {url}: {e}")
        return ""


def format_confidence_bar(confidence: float, length: int = 10) -> str:
    """Визуальная шкала уверенности."""
    filled = round(confidence * length)
    return "█" * filled + "░" * (length - filled)


def format_response(label: str, confidence: float, model_type: str, source: str = "") -> str:
    """Формирует ответ бота на армянском языке."""
    emoji = CATEGORY_EMOJI.get(label, "📰")
    bar = format_confidence_bar(confidence)
    pct = confidence * 100

    lines = [
        f"{emoji} *Այս լուրը պատկանում է՝ {label}*",
        "",
        f"📊 Վստահություն: `{bar}` {pct:.1f}%",
    ]

    if confidence < 0.65:
        lines.append("\n⚠️ _Ցածր վստահություն — արդյունքը կարող է ճշգրիտ չլինել_")
    elif confidence > 0.90:
        lines.append("\n✅ _Բարձր վստահություն_")

    if source:
        lines.append(f"\n🔗 Աղբյուր: {source}")

    return "\n".join(lines)


# ── Обработчики ───────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Բարև ձեզ\\!*\n\n"
        "Ես կարող եմ դասակարգել հայկական լրատվական հոդվածները՝\n"
        "• ⚽ *Սպորտ*\n"
        "• 🏛️ *Քաղաքական*\n\n"
        "Ուղղակի ուղարկեք ինձ *հոդվածի տեքստ* կամ *հղում*\\.\n\n"
        "/help — օգնություն\n"
        "/stats — մոդելի վիճակագրություն"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📖 *Օգնություն*\n\n"
        "Ինչպես օգտագործել բոտը:\n\n"
        "1️⃣ Ուղարկեք հոդվածի *տեքստ* \\(նվազագույնը 20 նիշ\\)\n"
        "2️⃣ Կամ ուղարկեք *հղում* azatutyun\\.am\\-ի հոդվածի վրա\n\n"
        "Բոտը կորոշի կատեգորիան և ցույց կտա վստահության մակարդակը\\."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pipeline, le, model_type = context.bot_data.get("model_info", (None, None, "N/A"))
    classes = list(le.classes_) if le else []
    text = (
        f"📈 *Մոդելի վիճակագրություն*\n\n"
        f"Մոդելի տեսակ: `{model_type}`\n"
        f"Կատեգորիաներ: {', '.join(classes)}\n"
        f"Լեզու: Հայերեն 🇦🇲\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главный обработчик сообщений."""
    user_text = update.message.text.strip()
    user = update.effective_user
    logger.info(f"Сообщение от @{user.username or user.id}: {user_text[:80]}")

    pipeline = context.bot_data.get("pipeline")
    le = context.bot_data.get("le")
    model_type = context.bot_data.get("model_type", "N/A")

    if pipeline is None or le is None:
        await update.message.reply_text(
            "❌ Մոդելը բեռնված չէ: Խնդրում ենք կապ հաստատել ադմինիստրատորի հետ:"
        )
        return

    # Статус «печатает»
    await update.message.chat.send_action("typing")

    # Если URL — скачиваем статью
    source_url = ""
    if is_url(user_text):
        source_url = user_text
        await update.message.reply_text("🔄 Հոդվածը բեռնվում է...")
        text_to_classify = await asyncio.get_event_loop().run_in_executor(
            None, fetch_article_text, user_text
        )
        if not text_to_classify or len(text_to_classify) < MIN_TEXT_LEN:
            await update.message.reply_text(
                "❌ Հղումից հոդված ստանալ հնարավոր չեղավ: "
                "Փորձեք ուղարկել հոդվածի տեքստը ուղղակիորեն:"
            )
            return
    else:
        text_to_classify = user_text

    if len(text_to_classify) < MIN_TEXT_LEN:
        await update.message.reply_text(
            f"⚠️ Տեքստը շատ կարճ է (նվազ. {MIN_TEXT_LEN} նիշ): "
            "Ուղղակի ուղարկեք ամբողջ հոդվածը:"
        )
        return

    # Классификация
    try:
        label, confidence = predict_text(text_to_classify, pipeline, le)
        response = format_response(label, confidence, model_type, source=source_url)
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"  → {label} ({confidence:.2%})")

    except Exception as e:
        logger.error(f"Ошибка классификации: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Դասակարգման ժամանակ սխալ տեղի ունեցավ: Փորձեք կրկին:"
        )


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок."""
    logger.error("Unhandled exception:", exc_info=context.error)


# ── Запуск ────────────────────────────────────────────────────────────────────

def main():
    if not BOT_TOKEN:
        logger.error(
            "Токен бота не задан! "
            "Установите переменную окружения: TELEGRAM_BOT_TOKEN=<ваш_токен>"
        )
        sys.exit(1)

    # Загружаем модель
    try:
        pipeline, le, model_type = load_model_safe()
    except FileNotFoundError:
        logger.error(
            "Модель не найдена! Сначала запустите: python train.py"
        )
        sys.exit(1)

    logger.info(f"Запускаем бот с моделью: {model_type}")

    # Создаём приложение
    app = Application.builder().token(BOT_TOKEN).build()

    # Сохраняем модель в bot_data
    app.bot_data["pipeline"] = pipeline
    app.bot_data["le"] = le
    app.bot_data["model_type"] = model_type
    app.bot_data["model_info"] = (pipeline, le, model_type)

    # Регистрируем хендлеры
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_error)

    logger.info("Бот запущен. Ожидаем сообщения...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
