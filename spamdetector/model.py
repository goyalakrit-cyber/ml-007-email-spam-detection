"""Text normalization, reproducible training, metrics and error review."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data import validate_frame

_SPECIAL_CHARACTERS = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase text, remove special characters, and normalize whitespace."""
    text = _SPECIAL_CHARACTERS.sub(" ", str(text).lower())
    return _WHITESPACE.sub(" ", text).strip()


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )


def train_model(
    frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> dict[str, Any]:
    """Train a vectorizer and classifier on a stratified reproducible split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    data = validate_frame(frame)
    if data["label"].value_counts().min() < 2:
        raise ValueError("Each class needs at least two messages for a stratified split.")
    train_text, test_text, train_labels, test_labels = train_test_split(
        data["text"],
        data["label"],
        test_size=test_size,
        random_state=seed,
        stratify=data["label"],
    )
    model = build_pipeline()
    model.fit(train_text, train_labels)
    predictions = model.predict(test_text)
    return {
        "model": model,
        "test_text": test_text,
        "test_labels": test_labels,
        "predictions": predictions,
        "seed": seed,
        "test_size": test_size,
    }


def evaluate_model(result: dict[str, Any]) -> dict[str, Any]:
    actual, predicted = result["test_labels"], result["predictions"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=["spam"],
        average="binary",
        pos_label="spam",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=["ham", "spam"]
        ),
    }


def error_analysis(result: dict[str, Any]) -> pd.DataFrame:
    """Return held-out mistakes and identify their false-positive/negative type."""
    errors = pd.DataFrame(
        {
            "text": result["test_text"],
            "actual": result["test_labels"],
            "predicted": result["predictions"],
        }
    )
    errors = errors[errors["actual"] != errors["predicted"]].copy()
    errors["error_type"] = errors.apply(
        lambda row: (
            "False positive" if row["predicted"] == "spam" else "False negative"
        ),
        axis=1,
    )
    return errors.reset_index(drop=True)
