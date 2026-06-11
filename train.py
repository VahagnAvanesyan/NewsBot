"""
train.py — Обучение модели классификации текста (армянский)
Pipeline: TF-IDF + Logistic Regression (baseline) / XGBoost (advanced)
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from utils import setup_logging, load_dataset, preprocess_for_ml, combine_title_text, print_dataset_stats

logger = setup_logging("train")

# ── Конфиг ────────────────────────────────────────────────────────────────────

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BASELINE_PATH = MODEL_DIR / "baseline_pipeline.pkl"
XGBOOST_PATH  = MODEL_DIR / "xgboost_pipeline.pkl"
ENCODER_PATH  = MODEL_DIR / "label_encoder.pkl"
LABELS_PATH   = MODEL_DIR / "labels.txt"

RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ── TF-IDF конфиг ─────────────────────────────────────────────────────────────
#
# Для армянского языка используем char-level n-grams (2-4):
#   • не нужен стемминг
#   • хорошо работает на морфологически богатых языках
#   • устойчив к опечаткам

TFIDF_PARAMS = {
    "analyzer":     "char_wb",   # символьные n-граммы с учётом границ слов
    "ngram_range":  (2, 4),
    "max_features": 100_000,
    "sublinear_tf": True,
    "min_df":       2,
    "strip_accents": None,       # не трогаем армянские символы
}


# ── Загрузка и подготовка данных ──────────────────────────────────────────────

def prepare_data(csv_path: str = "azatutyun_news.csv"):
    """Загружает датасет и готовит X, y."""
    df = load_dataset(csv_path)
    print_dataset_stats(df)

    # Объединяем заголовок (x3) + текст
    X_raw = combine_title_text(df)
    X = X_raw.apply(preprocess_for_ml)

    # Кодируем метки
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    logger.info(f"Классы: {list(le.classes_)}")
    logger.info(f"Размер датасета: {len(X)} примеров")

    return X, y, le, df


# ── Baseline: TF-IDF + Logistic Regression ────────────────────────────────────

def train_baseline(X_train, X_test, y_train, y_test, le: LabelEncoder):
    logger.info("═══ Baseline: TF-IDF + Logistic Regression ═══")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    # Кросс-валидация на train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
    logger.info(f"CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Обучаем на полном train
    pipeline.fit(X_train, y_train)

    # Оцениваем на test
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test F1-macro: {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    print_confusion_matrix(confusion_matrix(y_test, y_pred), le.classes_)

    return pipeline, acc, f1


# ── Advanced: TF-IDF + XGBoost ───────────────────────────────────────────────

def train_xgboost(X_train, X_test, y_train, y_test, le: LabelEncoder):
    try:
        import xgboost as xgb
        from scipy.sparse import issparse
    except ImportError:
        logger.warning("xgboost не установлен. Пропускаем.")
        return None, 0.0, 0.0

    logger.info("═══ Advanced: TF-IDF + XGBoost ═══")

    # XGBoost не поддерживает sparse напрямую в Pipeline — используем DMatrix
    # Поэтому строим TF-IDF отдельно
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    n_classes = len(le.classes_)
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    num_class = {} if n_classes == 2 else {"num_class": n_classes}

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        
        eval_metric="mlogloss",
        objective=objective,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **num_class,
    )
    model.fit(
        X_train_tfidf, y_train,
        eval_set=[(X_test_tfidf, y_test)],
        verbose=50,
    )

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    logger.info(f"XGBoost Test Accuracy: {acc:.4f}")
    logger.info(f"XGBoost Test F1-macro: {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    print_confusion_matrix(confusion_matrix(y_test, y_pred), le.classes_)

    # Упаковываем в объект с методом predict/predict_proba
    class XGBPipeline:
        def __init__(self, vec, clf, classes):
            self.vectorizer = vec
            self.classifier = clf
            self.classes_ = classes

        def predict(self, texts):
            X = self.vectorizer.transform(texts)
            return self.classifier.predict(X)

        def predict_proba(self, texts):
            X = self.vectorizer.transform(texts)
            p = self.classifier.predict_proba(X)
            if p.ndim == 1:   # binary
                return np.vstack([1 - p, p]).T
            return p

    pipeline_obj = XGBPipeline(vectorizer, model, le.classes_)
    return pipeline_obj, acc, f1


# ── Вспомогательное ───────────────────────────────────────────────────────────

def print_confusion_matrix(cm, classes):
    print("\nМатрица ошибок:")
    header = "          " + "  ".join(f"{c[:10]:>10}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{classes[i][:10]:>10}" + "  ".join(f"{v:>10}" for v in row)
        print(row_str)
    print()


def save_model(pipeline, path: Path, le: LabelEncoder):
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for cls in le.classes_:
            f.write(cls + "\n")
    logger.info(f"Модель сохранена: {path}")


def load_model(use_xgboost: bool = False):
    """Загружает сохранённую модель и энкодер."""
    model_path = XGBOOST_PATH if use_xgboost else BASELINE_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return pipeline, le


# ── Предсказание ──────────────────────────────────────────────────────────────

def predict_text(text: str, pipeline, le: LabelEncoder) -> tuple[str, float]:
    """
    Классифицирует текст.
    Возвращает (метка_на_армянском, уверенность).
    """
    processed = preprocess_for_ml(text)
    proba = pipeline.predict_proba([processed])[0]
    class_idx = int(np.argmax(proba))
    label = le.inverse_transform([class_idx])[0]
    confidence = float(proba[class_idx])
    return label, confidence


# ── Главная функция ───────────────────────────────────────────────────────────

def run_training(
    csv_path: str = "azatutyun_news.csv",
    use_xgboost: bool = True,
):
    """Полный цикл обучения."""
    logger.info("Загружаем датасет...")
    X, y, le, df = prepare_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Baseline всегда
    baseline_pipeline, bl_acc, bl_f1 = train_baseline(X_train, X_test, y_train, y_test, le)
    save_model(baseline_pipeline, BASELINE_PATH, le)

    best_pipeline = baseline_pipeline
    best_path = BASELINE_PATH

    # XGBoost (опционально)
    if use_xgboost:
        xgb_pipeline, xgb_acc, xgb_f1 = train_xgboost(X_train, X_test, y_train, y_test, le)
        if xgb_pipeline is not None:
            save_model(xgb_pipeline, XGBOOST_PATH, le)
            if xgb_f1 > bl_f1:
                best_pipeline = xgb_pipeline
                best_path = XGBOOST_PATH
                logger.info("✅ XGBoost лучше Baseline — используем XGBoost")
            else:
                logger.info("✅ Baseline лучше или равен — используем Baseline")

    # Тест на примерах
    logger.info("\n═══ Тест предсказаний ═══")
    test_samples = [
        "Հայաստանի հավաքականը հաղթեց ֆուտբոլային խաղում",   # спорт
        "Կառավարությունը նոր որոշում կայացրեց",              # политика
        "Ռոնալդոն գոլ խփեց Եվրոպայի չեմպիոնությունում",     # спорт
        "Ազգային ժողովն ընդունեց նոր օրենք",                # политика
    ]
    for sample in test_samples:
        label, conf = predict_text(sample, best_pipeline, le)
        logger.info(f"  «{sample[:40]}…»  → {label} ({conf:.2%})")

    logger.info("═══ Обучение завершено ═══")
    return best_pipeline, le


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="azatutyun_news.csv")
    parser.add_argument("--no-xgb",  action="store_true", help="Только Baseline")
    args = parser.parse_args()
    run_training(csv_path=args.csv, use_xgboost=not args.no_xgb)
