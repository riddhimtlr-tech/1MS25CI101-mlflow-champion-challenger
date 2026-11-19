
import json
import os
import platform
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "eval_inputs.csv"
OUTPUTS = ROOT / "outputs"
PLOTS = ROOT / "plots"

MODEL_A = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B = "siebert/sentiment-roberta-large-english"

SEED = 42
BATCH_SIZE = 16
MAX_LENGTH = 256

# Student USN
STUDENT_USN = "1MS25CI101"


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("champion-challenger-sentiment")

    df = pd.read_csv(DATA)
    df["label"] = df["label"].str.lower().str.strip()
    df["text"] = df["text"].fillna("").astype(str)

    if set(df["label"].unique()) != {"positive", "negative"}:
        raise ValueError(
            "Expected positive and negative reference labels only."
        )

    y_true = df["label"].tolist()
    texts = df["text"].tolist()

    device = 0 if torch.cuda.is_available() else -1
    results = []

    configs = [
        ("A", MODEL_A, "A_sentiment_distilbert"),
        ("B", MODEL_B, "B_sentiment_siebert"),
    ]

    for role, model_id, run_name in configs:
        print(f"\nLoading model {role}: {model_id}")

        classifier = pipeline(
            "sentiment-analysis",
            model=model_id,
            device=device,
        )

        start = time.perf_counter()
        raw = classifier(
            texts,
            batch_size=BATCH_SIZE,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        elapsed = time.perf_counter() - start

        predictions = []

        for item in raw:
            label = item["label"].lower()

            if label in {"positive", "label_1"}:
                predictions.append("positive")
            elif label in {"negative", "label_0"}:
                predictions.append("negative")
            else:
                raise ValueError(
                    f"Unexpected model label: {item['label']}"
                )

        accuracy = accuracy_score(y_true, predictions)

        macro_f1 = f1_score(
            y_true,
            predictions,
            labels=["negative", "positive"],
            average="macro",
        )

        weighted_f1 = f1_score(
            y_true, predictions, average="weighted"
        )

        precision = precision_score(
            y_true,
            predictions,
            pos_label="positive",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            pos_label="positive",
            zero_division=0,
        )

        latency_ms = elapsed * 1000 / len(texts)

        prediction_df = df.copy()
        prediction_df["prediction"] = predictions
        prediction_df["confidence"] = [
            float(item["score"]) for item in raw
        ]

        prediction_path = OUTPUTS / f"predictions_{role}.csv"
        prediction_df.to_csv(prediction_path, index=False)

        # Confusion matrix
        cm = confusion_matrix(
            y_true,
            predictions,
            labels=["negative", "positive"],
        )

        fig, ax = plt.subplots()
        image = ax.imshow(cm)

        ax.set(
            title=f"Confusion Matrix - Model {role}",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=[0, 1],
            xticklabels=["negative", "positive"],
            yticks=[0, 1],
            yticklabels=["negative", "positive"],
        )

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center"
                )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        confusion_path = PLOTS / f"confusion_matrix_{role}.png"
        fig.savefig(confusion_path, dpi=150)
        plt.close(fig)

        metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "precision_positive": float(precision),
            "recall_positive": float(recall),
            "latency_seconds": float(elapsed),
            "latency_per_sample_ms": float(latency_ms),
            "eval_samples": len(texts),
        }

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({
                "role": role,
                "task": "sentiment",
                "hf_model_id": model_id,
                "dataset_name": (
                    "news_sentiment_analysis_binary_balanced"
                ),
                "student_usn": STUDENT_USN,
                "python_version": platform.python_version(),
                "pandas_version": pd.__version__,
                "sklearn_version": sklearn.__version__,
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
            })

            mlflow.log_params({
                "hf_model_id": model_id,
                "task": "sentiment",
                "role": role,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "eval_samples": len(texts),
                "seed": SEED,
                "device": "cuda" if device >= 0 else "cpu",
            })

            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(prediction_path))
            mlflow.log_artifact(str(confusion_path))

        results.append({
            "role": role,
            "model_id": model_id,
            "run_id": run.info.run_id,
            **metrics,
        })

        print(
            f"Model {role} metrics:",
            json.dumps(metrics, indent=2),
        )

    # Save comparison summary
    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUTS / "metrics_summary.csv", index=False)

    with open(
        OUTPUTS / "metrics_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    # Compare macro-F1 and accuracy
    fig, ax = plt.subplots()
    x = np.arange(len(results))
    width = 0.35

    ax.bar(
        x - width / 2,
        [r["macro_f1"] for r in results],
        width,
        label="Macro-F1",
    )
    ax.bar(
        x + width / 2,
        [r["accuracy"] for r in results],
        width,
        label="Accuracy",
    )

    ax.set_xticks(x, [r["role"] for r in results])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Champion-Challenger Metric Comparison")
    ax.legend()
    fig.tight_layout()

    comparison_plot = PLOTS / "metric_comparison.png"
    fig.savefig(comparison_plot, dpi=150)
    plt.close(fig)

    # Log comparison run
    with mlflow.start_run(run_name="compare_sentiment"):
        mlflow.set_tags({
            "role": "comparison",
            "task": "sentiment",
            "dataset_name": (
                "news_sentiment_analysis_binary_balanced"
            ),
            "student_usn": STUDENT_USN,
        })

        mlflow.log_params({
            "primary_metric": "macro_f1",
            "tie_breaker": "accuracy",
            "latency_guardrail_ms_per_sample": 1000,
        })

        for row in results:
            mlflow.log_metrics({
                f"{row['role']}_accuracy": row["accuracy"],
                f"{row['role']}_macro_f1": row["macro_f1"],
                f"{row['role']}_latency_per_sample_ms":
                    row["latency_per_sample_ms"],
            })

        mlflow.log_artifact(
            str(OUTPUTS / "metrics_summary.csv")
        )
        mlflow.log_artifact(
            str(OUTPUTS / "metrics_summary.json")
        )
        mlflow.log_artifact(str(comparison_plot))

    print("\nComparison summary:")
    print(
        summary[
            [
                "role",
                "model_id",
                "accuracy",
                "macro_f1",
                "latency_per_sample_ms",
            ]
        ].to_string(index=False)
    )

    print("\nEvaluation finished. View results with: mlflow ui")


if __name__ == "__main__":
    main()
