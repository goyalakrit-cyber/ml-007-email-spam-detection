import unittest

import pandas as pd

from spamdetector.data import validate_frame
from spamdetector.model import clean_text, error_analysis, evaluate_model, train_model


SAMPLE = pd.DataFrame(
    {
        "label": ["ham", "spam"] * 8,
        "text": [
            "Hey are we meeting today?",
            "WIN cash prize click now",
            "Can you call me later",
            "Claim your free offer now",
            "Lunch at the usual place",
            "Urgent winner collect your prize",
            "Thanks for your help",
            "Free entry reply to win",
            "See you at the station",
            "Congratulations claim cash",
            "Please send the notes",
            "You have won a free gift",
            "Dinner is ready at home",
            "Get your reward by clicking",
            "I will call after work",
            "Winner claim your cash now",
        ],
    }
)


class ModelTests(unittest.TestCase):
    def test_text_cleaning(self):
        self.assertEqual(clean_text("  WIN!!!  Cash\tNOW. "), "win cash now")

    def test_dataset_schema(self):
        with self.assertRaisesRegex(ValueError, "label and text"):
            validate_frame(pd.DataFrame({"message": ["hello"]}))

    def test_training_is_reproducible_and_evaluated(self):
        first = train_model(SAMPLE, test_size=0.25, seed=7)
        second = train_model(SAMPLE, test_size=0.25, seed=7)
        self.assertEqual(first["test_text"].tolist(), second["test_text"].tolist())
        self.assertEqual(first["predictions"].tolist(), second["predictions"].tolist())
        metrics = evaluate_model(first)
        self.assertEqual(metrics["confusion_matrix"].shape, (2, 2))
        self.assertTrue(0 <= metrics["f1"] <= 1)

    def test_error_analysis_marks_both_error_types(self):
        result = {
            "test_text": pd.Series(["clean", "bad"]),
            "test_labels": pd.Series(["ham", "spam"]),
            "predictions": ["spam", "ham"],
        }
        mistakes = error_analysis(result)
        self.assertEqual(set(mistakes["error_type"]), {"False positive", "False negative"})


if __name__ == "__main__":
    unittest.main()
