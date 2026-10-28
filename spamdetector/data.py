"""Acquire and validate the publicly available UCI SMS Spam Collection."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_FILENAME = "SMSSpamCollection"
EXAMPLE_MESSAGES = (
    ("Ham", "Hey, are you coming to class today?"),
    ("Spam", "Congratulations! You have won a free prize."),
)


def load_dataset(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load cached UCI data or download and extract it once."""
    directory = Path(data_dir)
    path = directory / DATASET_FILENAME
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                member = next(
                    (item for item in archive.namelist()
                     if Path(item).name == DATASET_FILENAME),
                    None,
                )
                if member is None:
                    raise ValueError("UCI archive did not contain SMSSpamCollection.")
                path.write_bytes(archive.read(member))
        except zipfile.BadZipFile as exc:
            raise ValueError("UCI returned an invalid dataset archive.") from exc

    frame = pd.read_csv(
        path, sep="\t", header=None, names=["label", "text"], encoding="utf-8"
    )
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame[frame["label"].isin(["ham", "spam"])].reset_index(drop=True)
    if frame.empty:
        raise ValueError("The dataset contains no valid ham/spam rows.")
    return frame


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate labelled text data and keep only usable ham/spam examples."""
    missing = {"label", "text"} - set(frame.columns)
    if missing:
        raise ValueError(
            "Dataset must contain label and text columns; missing "
            + ", ".join(sorted(missing))
            + "."
        )
    clean = frame.loc[:, ["label", "text"]].copy()
    clean["label"] = clean["label"].astype(str).str.strip().str.lower()
    clean["text"] = clean["text"].fillna("").astype(str)
    clean = clean[clean["label"].isin(["ham", "spam"])]
    clean = clean[clean["text"].str.strip().ne("")]
    if clean["label"].nunique() != 2:
        raise ValueError("Dataset must include at least one ham and one spam message.")
    return clean.reset_index(drop=True)
