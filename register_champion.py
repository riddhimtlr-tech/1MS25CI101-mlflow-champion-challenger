
import json
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient
from transformers import pipeline

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "outputs" / "metrics_summary.json"

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "SentimentChampionChallenger"

MODEL_A = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B = "siebert/sentiment-roberta-large-english"

STUDENT_USN = "1MS25CI101"
LATENCY_GUARDRAIL_MS = 1000.0


class HuggingFaceSentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from transformers import pipeline

        model_id = context.model_config["hf_model_id"]
        device = 0 if torch.cuda.is_available() else -1

        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_id,
            device=device,
        )

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].fillna("").astype(str).tolist()

        outputs = self.classifier(
            texts,
            truncation=True,
            max_length=256,
        )

        results = []
        for item in outputs:
            label = item["label"].lower()

            if label in {"positive", "label_1"}:
                label = "positive"
            elif label in {"negative", "label_0"}:
                label = "negative"

            results.append({
                "prediction": label,
                "confidence": float(item["score"]),
            })

        return pd.DataFrame(results)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    with open(SUMMARY_PATH, encoding="utf-8") as f:
        results = json.load(f)

    # Primary metric: macro-F1.
    # Tie-breaker: accuracy.
    # Guardrail: latency must be at most 1000 ms/sample.
    eligible = [
        row for row in results
        if row["latency_per_sample_ms"] <= LATENCY_GUARDRAIL_MS
    ]

    if not eligible:
        raise RuntimeError("No model passed the latency guardrail.")

    eligible.sort(
        key=lambda row: (row["macro_f1"], row["accuracy"]),
        reverse=True,
    )

    champion = eligible[0]
    challenger = next(
        (
            row for row in results
            if row["model_id"] != champion["model_id"]
        ),
        None,
    )

    if challenger is None:
        raise RuntimeError("Could not identify the challenger model.")

    # Register both models with the same registered model name.
    for role, row in [
        ("champion", champion),
        ("challenger", challenger),
    ]:
        model_id = row["model_id"]

        print(f"\nRegistering {role}: {model_id}")

        with mlflow.start_run(run_name=f"register_{role}") as run:
            mlflow.set_tags({
                "student_usn": STUDENT_USN,
                "role": role,
                "hf_model_id": model_id,
                "macro_f1": str(row["macro_f1"]),
                "accuracy": str(row["accuracy"]),
            })

            mlflow.log_params({
                "hf_model_id": model_id,
                "registered_model_name": REGISTERED_MODEL_NAME,
            })

            mlflow.pyfunc.log_model(
                name="sentiment_model",
                python_model=HuggingFaceSentimentModel(),
                model_config={"hf_model_id": model_id},
                pip_requirements=[
                    "mlflow",
                    "pandas",
                    "torch",
                    "transformers",
                ],
                registered_model_name=REGISTERED_MODEL_NAME,
            )

        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

        matching = [
            version for version in versions
            if version.run_id == run.info.run_id
        ]

        if not matching:
            raise RuntimeError(
                f"Could not find registered version for {model_id}"
            )

        version = max(matching, key=lambda v: int(v.version))

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version.version,
            "hf_model_id",
            model_id,
        )
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version.version,
            "student_usn",
            STUDENT_USN,
        )

        alias = role
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            alias,
            version.version,
        )

        print(
            f"Assigned alias '{alias}' to version {version.version}"
        )

    print("\nRegistration completed.")
    print(f"Registered model: {REGISTERED_MODEL_NAME}")
    print(f"Champion: {champion['model_id']}")
    print(f"Challenger: {challenger['model_id']}")


if __name__ == "__main__":
    main()
