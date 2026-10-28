"""Command-line training and evaluation entry point."""

import argparse

from .data import load_dataset
from .model import error_analysis, evaluate_model, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the UCI SMS spam baseline.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_model(load_dataset(), test_size=args.test_size, seed=args.seed)
    metrics = evaluate_model(result)
    print(f"Split: {1 - args.test_size:.0%} train / {args.test_size:.0%} test; seed={args.seed}")
    for metric in ("accuracy", "precision", "recall", "f1"):
        print(f"{metric.title():>10}: {metrics[metric]:.4f}")
    print("\nConfusion matrix (rows=actual ham,spam; columns=predicted ham,spam):")
    print(metrics["confusion_matrix"])
    errors = error_analysis(result)
    print(f"\nHeld-out errors: {len(errors)}")
    if not errors.empty:
        print(errors.groupby("error_type").size().to_string())


if __name__ == "__main__":
    main()
