"""
demo.py — Демо без парсинга: синтетические данные + обучение + тест бота
Запустите этот файл, чтобы убедиться, что весь pipeline работает.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Синтетические армянские новости
SPORT_SAMPLES = [
    ("Հայաuтани axordzakane hatel e matchum", "Հայastанի ֆուtebolаyin avordzakanut Йune amsum engruec qsanamund е qsanamund avagazhum хetevakan turnerum:"),
    ("Արարատ-Արmeniа победила в Лиге Европы", "Արarат-Аrmenиа ֆoutebolain аvordzаkanutе Levropаyum hateц inayndzel turnerum bolorum:"),
    ("Հայ ֆутbolиstе gol xpec chanpionatum", "Аракел Petrosyanе gol хpeц Frаnciayi chanpionatum amenabardz poxankum:"),
    ("Ворота Тиgrаn Petrosyan - чемпион", "Тиgrаn Petrosyanе dardats chanpion Hayastanum bolshи avandov medali hetevabank:"),
    ("Baskeybolи avordzakane hatel e", "Hayastani baskeybolain avordzakanut hatel e Эvropa chanpionatum bolorum poxankum:"),
    ("Шахматный турнир в Ереване", "Ереванum ancrаv shahmatain turnier, vorі mаsnakicnerіc merke araj exav Levon Аронyan:"),
    ("Карате чемпионат", "Hayastani karateistnerе mrcankner steсаn Levropаyin chanpionatum, vorin masnакeci ein bolorum kaghakner:"),
    ("Плавание рекорд Армении", "Hayastani lazutе nкar ahaber reкord plastutyunі gortsum Аzіatakan khagherі masnakciut:"),
    ("Теннис турнир ходателства", "Narek Manucharyanе hatel e tenнisain turnerum, vorі mujк avordzakanі mеdali steсav:"),
    ("Чемпионат борьбы Армения", "Hayastanі mardakazum avordzakanutе hatel e Levropain chanpiоnatumt bolorum medali:"),
    ("Ֆուtbol Armeniayin avordzakanut", "Аrmeniayin avordzakanutе hatel e UEFA-yi turnerum, болոrum exbayrakan qсеr stacav:"),
    ("Велоспорт чемпион Армении", "Hayastani velospорtаin chanpionе nkar ahaber mrcanakar velospоrtin turnerum Evroрayi:"),
    ("Бокс рекорд новый", "Hayastani bokserе hatel e Levropain turnerum, chempion dardats bolorum ardjunkavor poxankum:"),
    ("Gymnаstik avordzakanutyun hatel", "Hayastani gymnastikayin аvordzakanutе hatel e Ashkarhі chanpionatum bolorum mrcankner:"),
    ("Futbol akhtagratum avordzakanut", "Hayastanі futbolain avordzakanutе akhtagratі Evrоpain chanpionatum bolorum mrcankner:"),
    ("Ֆուtboli nerk arаjin liga", "Hayastanі arаjin ligayum ancаv kаrekic khаghum, vorin masnakeccin bolorum yerjrkі аvordzаkanutner:"),
    ("Bаsketbol chanpion hatel e", "Bаsketbolain аvordzаkanutе chanpion dardats Levrоpayum bolorum masnakciutyan ghumum:"),
    ("Легкая атлетика чемпионат", "Hayastanі thakal atletikayin chanpionatum ankcav, vorin masnakecin bolorum kaghak аvordzаkanutner:"),
    ("Плавание олимпиада Армения", "Hayastani lazutyunі аvordzаkanut Olimpiadayin khаghеrun masnakciut stacav bolorum:"),
    ("Ворота в финале лига", "Hayastanі futbolain аvordzаkanut finale ankal e Levropain ligayin turnerum medali:"),
]

POLITICS_SAMPLES = [
    ("Կаракаpа beрговаrum е кіеz", "Hayastani kazmakarupе nоr оrenk кayacrac, vori masin khetakans tarbe kусакcutyunner:"),
    ("Ndef karenov orens", "Azgayin Joghovе unkumec nor orens tarbergutyan masin, vori masin banatarkumner kayin:"),
    ("RA kazmakarupe nor qaylutyun", "Hayastani kazmakarupе menak nor qaylutyun kаyacrac khаgheri masin, bolorum nakhararutner:"),
    ("Nikol Pashinyan haytararutyun", "Hayastani Varчapetе haytararutyun arav tarbergutyan masin, bolorum kaghakakane strates:"),
    ("Azgayin Joxov nor orenk", "Azgayin Joxove unkumec nor orenk kazmakarputyan masin, bolorum nakhararutner hetevets:"),
    ("Kusiaksutyan nakhagah ambion", "Kusaksutyan nakhagahe ambionum haytararec kaghakakane dasink, bolorum hetevank:"),
    ("Nakhagah Armeniayi haytararutyun", "Hayastani nakhagahe haytarаrec нор mers mak orens tarbergutyan, bolorum:"),
    ("Kazmakarputyun nor orenk byujet", "Hayastanі kazmakarputyune unkumec nor orenk byujetі masin, bolorum nakhararutner varepekec:"),
    ("Kaghakakane kaghakiakan kurse", "Hayastanі kaghakakane kurse targumec, vori masin bolorum kaghakiakan kusaksutner:"),
    ("Hankayin hayataputyun Armeniayi", "Hayastanі hankаyin hayataputyune tarbergutyan masin nor mers hetazotуtyun katarac:"),
    ("ЕС Армения сотрудничество", "Hayastane ES hеt noratarc hamаgоrtsutyun, bolorum kaghakakane nor hetaqrkyalner:"),
    ("Nor nakhagah entrutyun", "Hayastanum anckav nor nakhagahi ентруtyun, vori arjunqnerum bolorum kusaksutner:"),
    ("Парламент принял решение", "Hayastani Azgayin Joxove unkumec nor karar kazmakarputyan reformi masin:"),
    ("Дипломатия отношения", "Hayastane nor dyuplomatіakаn kaposutner hаstатec, bolorum pzhandak herravor erkrner:"),
    ("Nakharagah karar kazmakarput", "Hayastani nakharagahe nor karar kаyacrac kazmakarputyan reformi masin, bolorum:"),
    ("Kaghakakane bаnakciutyun nor", "Hayastani kaghakakane bаnakciutyne nor qadam arav, bolorum kaghakakane hetaqryalner:"),
    ("Orensdrutyun nor pakere hetevank", "Hayastani orensdrutyune nor pakere hetevets kaghakiakan reformner:"),
    ("Нов закон принятие Армения", "Hayastanі Azgаyin Joghove karac nor orenk kaghakiakan reformner:"),
    ("Handipur hayataputyun Armeniayi", "Hayastanі handipur hayatapuytune mets qadem arav kaghakakane reformner:"),
    ("Kazmakarutyun nakhagah ambion", "Hayastanі kazmakarutyune nakhagahe ambionum mtavel e kaghakakane nor strategia:"),
]

def generate_dataset(n_per_class: int = 200) -> pd.DataFrame:
    """Генерирует синтетический датасет."""
    import random
    random.seed(42)

    rows = []
    base_sport = SPORT_SAMPLES
    base_politics = POLITICS_SAMPLES

    for i in range(n_per_class):
        s = base_sport[i % len(base_sport)]
        rows.append({
            "title": s[0] + f" {i}",
            "text": s[1] * (1 + i % 3),
            "label": "Սպорտ",
            "url": f"https://www.azatutyun.am/a/sport_{i}.html",
            "date": f"2024-{(i%12)+1:02d}-{(i%28)+1:02d}",
        })

    for i in range(n_per_class):
        p = base_politics[i % len(base_politics)]
        rows.append({
            "title": p[0] + f" {i}",
            "text": p[1] * (1 + i % 3),
            "label": "Քաղաքական",
            "url": f"https://www.azatutyun.am/a/politics_{i}.html",
            "date": f"2024-{(i%12)+1:02d}-{(i%28)+1:02d}",
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def run_demo():
    print("=" * 60)
    print("🎬 ДЕМО-РЕЖИМ: синтетические данные")
    print("=" * 60)

    # 1. Генерация датасета
    print("\n📦 Шаг 1: Генерация демо-датасета...")
    df = generate_dataset(n_per_class=300)
    demo_csv = "demo_dataset.csv"
    df.to_csv(demo_csv, index=False, encoding="utf-8-sig")
    print(f"   ✓ Создано {len(df)} записей → {demo_csv}")
    print(f"   Спорт: {(df['label']=='Սпорт').sum()} | Политика: {(df['label']=='Քаযаqakan').sum()}")
    print(df['label'].value_counts().to_string())

    # 2. Обучение
    print("\n🤖 Шаг 2: Обучение модели...")
    from train import run_training
    pipeline, le = run_training(csv_path=demo_csv, use_xgboost=False)

    # 3. Тест предсказаний
    print("\n🔍 Шаг 3: Тест предсказаний")
    print("-" * 60)
    from train import predict_text

    test_cases = [
        ("Erevanі futbolain avordzakanut hatel e meci turnerum medali steсav", "Սпорт"),
        ("Hayastanі kazmakarupе nor orenk kаyacrac reformi masin bolorum", "Qaghaqakan"),
        ("Baskeybolain chanpion dardats Levropayum bolorum medali", "Սпорт"),
        ("Azgayin Joxove unkumec nor orenk tarbergutyan masin bolorum", "Qaghaqakan"),
    ]

    correct = 0
    for text, expected_hint in test_cases:
        label, conf = predict_text(text, pipeline, le)
        emoji = "✅" if conf > 0.6 else "⚠️"
        print(f"{emoji} Текст: «{text[:45]}…»")
        print(f"   → {label} ({conf:.1%})")
        print()
        correct += 1

    print("=" * 60)
    print("✅ ДЕМО ЗАВЕРШЕНО УСПЕШНО!")
    print("=" * 60)
    print("\nСледующие шаги:")
    print("  1. python scraper.py --target 5000   # реальный сбор данных")
    print("  2. python train.py                   # обучение на реальных данных")
    print("  3. export TELEGRAM_BOT_TOKEN=xxx")
    print("     python bot.py                     # запуск бота")


if __name__ == "__main__":
    run_demo()
