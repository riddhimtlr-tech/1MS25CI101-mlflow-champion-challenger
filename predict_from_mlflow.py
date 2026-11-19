
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "new_data.csv"
OUTPUT_PATH = ROOT / "outputs" / "new_data_predictions.csv"

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "SentimentChampionChallenger"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@champion"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {DATA_PATH}\n"
            "Create data/new_data.csv with a column named 'text'."
        )

    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns:
        raise ValueError(
            "data/new_data.csv must contain a 'text' column."
        )

    if df.empty:
        raise ValueError("data/new_data.csv contains no examples.")

    print(f"Loading registered champion: {MODEL_URI}")
    model = mlflow.pyfunc.load_model(MODEL_URI)

    predictions = model.predict(df[["text"]])

    # Resolve the version behind the champion alias.
    version_info = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, "champion"
    )
    hf_model_id = version_info.tags.get(
        "hf_model_id", "not recorded"
    )

    output = df.copy()

    if isinstance(predictions, pd.DataFrame):
        output["prediction"] = predictions["prediction"].tolist()

        if "confidence" in predictions.columns:
            output["confidence"] = predictions["confidence"].tolist()
    else:
        output["prediction"] = predictions

    output["mlflow_model_name"] = REGISTERED_MODEL_NAME
    output["mlflow_model_version"] = version_info.version
    output["mlflow_alias"] = "champion"
    output["hf_model_id"] = hf_model_id

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\nPrediction results:")
    print(output.to_string(index=False))
    print(f"\nMLflow model version: {version_info.version}")
    print(f"Hugging Face model: {hf_model_id}")
    print(f"Saved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
