# Champion–Challenger Model Comparison with Hugging Face and MLflow

## Project Overview

This project compares two Hugging Face sentiment classification models using MLflow experiment tracking. The goal is to evaluate their predictive performance, compare latency, and register the selected champion and challenger models.

## Models Evaluated

- **Model A:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- **Model B:** `siebert/sentiment-roberta-large-english`

## Dataset

The evaluation dataset is stored in `data/eval_inputs.csv` and contains 300 balanced positive and negative examples sampled from the provided news sentiment dataset. Neutral examples were excluded because the selected models perform binary sentiment classification. The original dataset labels may contain noise, so the results should be interpreted with this limitation in mind.

## Evaluation Results

| Metric | Model A: DistilBERT | Model B: SieBERT |
|---|---:|---:|
| Accuracy | 57.33% | 59.67% |
| Macro-F1 | 57.18% | 58.26% |
| Latency per sample | 23.61 ms | 175.62 ms |

## Champion Selection and Decision

SieBERT was selected as the champion because it achieved the higher macro-F1 score, which was the primary selection metric. Its macro-F1 was approximately 1.08 percentage points higher than DistilBERT's. However, SieBERT was substantially slower, taking approximately 7.4 times as long per sample in this evaluation. DistilBERT remains the challenger because it offers lower latency. The performance difference is modest, and these results apply only to the evaluation sample used in this project.

## How to Run

Activate the virtual environment and install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Run evaluation, register the models, and generate predictions:

```bash
python evaluate.py
python register_champion.py
python predict_from_mlflow.py
```

The scripts use MLflow to track evaluation runs and register model versions. Generated prediction files and metrics are stored in `outputs/`, while plots are stored in `plots/`.

## Limitations

The evaluation uses a limited sample of news headlines and descriptions. Dataset labels may be imperfect, and the models' binary outputs do not cover neutral sentiment. Latency may vary with hardware, caching, and runtime conditions.

## AI Assistance Disclosure

AI assistance was used to support code development, debugging, project organization, and documentation. The reported metrics come from running the evaluation scripts in this project.
