🎯 What it does

Scrapes news from azatutyun.am and classifies them into two categories:

CategoryArmenianEmojiSportsՍպորտ⚽PoliticsՔաղաքական🏛️


📁 Project Structure

azatutyun_classifier/
├── scraper.py        # News scraper (requests + BeautifulSoup)
├── train.py          # Model training (TF-IDF + Logistic Regression)
├── bot.py            # Telegram bot (python-telegram-bot v20)
├── utils.py          # Helper functions
├── demo.py           # Demo without scraping
├── requirements.txt  # Dependencies
└── models/           # Saved models (after training)


⚙️ Installation

# 1. Clone the repository
git clone https://github.com/VahagnAvanesyan/NewsBot.git
cd NewsBot

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install requests beautifulsoup4 lxml pandas numpy scikit-learn xgboost python-telegram-bot


🚀 Quick Start

Step 1 — Test the pipeline (demo)

bashpython demo.py

Step 2 — Collect data

bashpython scraper.py --target 5000


⏱ Takes a few hours. If interrupted — resumes from checkpoint automatically.



Step 3 — Train the model

bashpython train.py --no-xgb

Step 4 — Run the bot

bash# Windows
$env:TELEGRAM_BOT_TOKEN="your_token_here"

# Mac/Linux
export TELEGRAM_BOT_TOKEN="your_token_here"

python bot.py


🤖 Telegram Bot

Create a bot via @BotFather using /newbot to get your token.

What the bot can do:

ActionResultSend news textClassification + confidence scoreSend article URLAuto-fetch + classification/startWelcome message/helpInstructions/statsModel info

Example response:

⚽ Այս լուրը պատկանում է՝ Սպորտ
📊 Confidence: ███████░░░ 73.8%


📊 Model Results

Trained on 215 real articles from azatutyun.am:

MetricValueAccuracy97.7%F1-macro97.7%Precision (Sports)100%Recall (Politics)100%


💡 Accuracy will improve further with 5000+ articles




🛠️ Tech Stack


Scraping: requests + BeautifulSoup4
ML: scikit-learn — TF-IDF (char n-grams) + Logistic Regression
Bot: python-telegram-bot v20 (async)
Language: Armenian 🇦🇲 (UTF-8)



👤 Author

Vahagn Avanesyan — github.com/VahagnAvanesyan


📄 License

MIT — free to use.
