# 📰 Azatutyun News Classifier

Система классификации армянских новостей с сайта [azatutyun.am](https://www.azatutyun.am) на две категории:
- ⚽ **Սпорт** (Спорт)
- 🏛️ **Qaghaqakan** (Политика)

---

## 📁 Структура проекта

```
azatutyun_classifier/
├── scraper.py          # Парсинг новостей с azatutyun.am
├── train.py            # Обучение ML-модели (TF-IDF + LR / XGBoost)
├── bot.py              # Telegram-бот
├── utils.py            # Вспомогательные функции
├── demo.py             # Демо без парсинга (синтетические данные)
├── requirements.txt    # Зависимости
├── models/             # Сохранённые модели (создаётся автоматически)
└── logs/               # Логи (создаётся автоматически)
```

---

## ⚙️ Установка

### 1. Клонируйте / скопируйте проект
```bash
cd azatutyun_classifier
```

### 2. Создайте виртуальное окружение
```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Установите зависимости
```bash
pip install -r requirements.txt
```

---

## 🚀 Быстрый старт (ДЕМО)

Если хотите сразу проверить, что всё работает — без парсинга:

```bash
python demo.py
```

Скрипт создаст синтетический датасет, обучит модель и покажет предсказания.

---

## 📊 Часть 1 — Сбор данных

```bash
# Собрать по 5000 статей каждой категории (занимает несколько часов)
python scraper.py --target 5000 --output azatutyun_news.csv

# Тест: собрать по 50 статей
python scraper.py --target 50 --output test_news.csv
```

### Как работает парсер

| Параметр | Значение |
|---|---|
| Категория «Спорт» | `/z/1517` |
| Категория «Политика» | `/z/1516` |
| Задержка между запросами | 1.5–3.5 сек (случайная) |
| Повторные попытки | 3 раза |
| Checkpoint | `checkpoint_Սпорт.csv`, `checkpoint_Qaghaqakan.csv` |

> ⚠️ **Если сбор прерван** — просто запустите снова. Парсер продолжит с checkpoint.

### Структура CSV (`azatutyun_news.csv`)

| Колонка | Описание |
|---|---|
| `title` | Заголовок статьи (армянский) |
| `text` | Полный текст статьи |
| `label` | `Սпорт` или `Qaghaqakan` |
| `url` | Прямая ссылка на статью |
| `date` | Дата публикации (`YYYY-MM-DD`) |

---

## 🤖 Часть 2 — Обучение модели

```bash
# Обучить с XGBoost (лучшее качество)
python train.py --csv azatutyun_news.csv

# Только Baseline (TF-IDF + Logistic Regression — быстрее)
python train.py --csv azatutyun_news.csv --no-xgb
```

После обучения в папке `models/` появятся:
- `baseline_pipeline.pkl` — TF-IDF + Logistic Regression
- `xgboost_pipeline.pkl` — TF-IDF + XGBoost
- `label_encoder.pkl` — энкодер меток
- `labels.txt` — список категорий

### Ожидаемые метрики (на 5000+ реальных статей)

| Модель | Accuracy | F1-macro |
|---|---|---|
| TF-IDF + LR | ~92–95% | ~92–95% |
| TF-IDF + XGBoost | ~94–97% | ~94–97% |

> 💡 Для армянского языка используются **символьные n-граммы (char_wb, 2–4)** — не нужны стеммеры и лемматизаторы.

---

## 🤖 Часть 3 — Telegram бот

### 1. Создайте бота у [@BotFather](https://t.me/BotFather)

```
/newbot → введите имя → получите токен
```

### 2. Установите токен

```bash
# Linux / macOS
export TELEGRAM_BOT_TOKEN="1234567890:AABBCCDDEEFFaabbccddeeff"

# Windows (PowerShell)
$env:TELEGRAM_BOT_TOKEN="1234567890:AABBCCDDEEFFaabbccddeeff"

# Windows (CMD)
set TELEGRAM_BOT_TOKEN=1234567890:AABBCCDDEEFFaabbccddeeff
```

### 3. Запустите бота

```bash
python bot.py
```

### Что умеет бот

| Команда / действие | Результат |
|---|---|
| `/start` | Приветствие |
| `/help` | Инструкция |
| `/stats` | Информация о модели |
| Текст новости | Классификация + confidence |
| Ссылка на azatutyun.am | Автозагрузка + классификация |

### Пример ответа

```
⚽ Այս լурը паtкаnum е՝ Спорт

📊 Вstаhutyun: ████████░░ 82.4%
```

---

## 🔄 Полный pipeline (от нуля до бота)

```bash
# Шаг 1: Собрать данные
python scraper.py --target 5000

# Шаг 2: Обучить модель
python train.py

# Шаг 3: Запустить бота
export TELEGRAM_BOT_TOKEN="ваш_токен"
python bot.py
```

---

## 🛠️ Troubleshooting

### Ошибка кодировки армянского текста
```python
# В CSV всегда используем utf-8-sig
df.to_csv("file.csv", encoding="utf-8-sig")
# При чтении
df = pd.read_csv("file.csv", encoding="utf-8-sig")
```

### Парсер возвращает пустые статьи
Сайт мог изменить вёрстку. Откройте `scraper.py` → функцию `parse_article_links()` и обновите CSS-селекторы.

### Бот не отвечает
1. Проверьте токен: `echo $TELEGRAM_BOT_TOKEN`
2. Убедитесь, что модель обучена: папка `models/` должна содержать `.pkl` файлы
3. Посмотрите лог: `logs/bot.log`

### XGBoost не устанавливается
```bash
pip install xgboost --upgrade
# или только baseline:
python train.py --no-xgb
```

---

## 📋 Требования

- Python 3.10+
- ОЗУ: минимум 4 GB (для датасета 10 000+ статей)
- Disk: ~500 MB для датасета + модели

---

## 📄 Лицензия

MIT — используйте свободно.
