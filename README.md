# ML-007 Email & SMS Spam Detection

A beginner-friendly text-classification project with an interactive Streamlit
website. Train and evaluate a reproducible TF-IDF + Multinomial Naive Bayes
model, inspect false positives and false negatives, and try predictions on
your own messages.

> **Dataset scope:** the included public dataset contains SMS messages, not
> email. It is a useful text-spam benchmark, but performance on real email may
> differ. The app identifies the dataset scope accurately.

## Features

- Downloads the UCI SMS Spam Collection; no account or credentials required.
- Automatically lowercases, removes special characters, normalizes whitespace,
  and tokenizes text before TF-IDF; the website previews cleaned text and tokens.
- Offers optional English stopword removal and stemming settings, applied the
  same way during training and interactive prediction.
- Uses a stratified, reproducible train/test split and Multinomial Naive Bayes.
- Reports accuracy, spam precision/recall/F1 and a confusion matrix.
- Displays held-out false positives and false negatives.
- Offers interactive predictions with estimated class confidence.
- Accepts compatible labelled CSV uploads (`label`, `text` columns).

## Quick start

Python 3.9 or later is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app downloads and caches the dataset under `data/` on first run. Source:
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
Citation: Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011).
Contributions to the study of SMS Spam Filtering: New Collection and Results.
Proceedings of the 2011 ACM Symposium on Document Engineering. See the UCI
dataset page for licensing details.

## Command-line evaluation

```bash
python -m spamdetector.train --test-size 0.2 --seed 42
```

Run the offline test suite with:

```bash
python -m unittest discover -s tests -v
```

## Limitations

The benchmark is SMS data, not email; performance on current email and other
message sources can differ. The dataset is not committed to this repository.
The first run requires internet access to UCI, but later runs use the cached
copy. Metrics are a baseline rather than a performance guarantee. Do not enter
sensitive messages into a shared deployment.