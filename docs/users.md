# User Guide

## 1. Environment Setup
- Requires Python 3.10+ and git.
- Create a virtual environment and install requirements:
  ```bash
  python -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  ```
- Update `config.yaml` paths if your data lives outside the repository.

## 2. Running the CLI Pipeline
- **Train from scratch**
  ```bash
  python main.py --config config.yaml
  ```
  Generates processed splits under `data/processed/`, saves checkpoints in `models/`, and writes run logs to `logs/`.
- **Reuse the latest promoted model**
  ```bash
  python main.py --skip-training --input <path/to/jobs.csv>
  ```
  Input CSV must contain `job_title` and `duties_description`. Results are written to `data/processed/predictions_<DATE>.csv`, leaving the source file unchanged.
- **Optional flags**
  - `--explain` adds attention-based explanation HTML alongside predictions.
  - `--force-update-best` overrides metric checks and replaces `models/best_model/` with the newly trained weights.

## 3. FastAPI Service
- Start the API with:
  ```bash
  python api_server.py
  ```
  Visit `http://localhost:8000/docs` for interactive endpoints.
- Key endpoints:
  - `POST /predict/job` — single JSON payload with `job_title` and `duties_description`.
  - `POST /predict/batch` — send `{ "jobs": [ ... ] }` to classify multiple records in one call.
  - `POST /predict/csv` — upload a CSV; receive a CSV with predictions.
- The service automatically loads `models/best_model/` and caches the artifacts in memory.

## 4. Artefact Management
- Best-performing weights live in `models/best_model/` and are consumed by both the CLI and API.
- Timestamped checkpoints for each training run are preserved under `models/checkpoint-final-*/`.
- Reference mappings and metadata are stored beside the model for portability (`label2id.json`, `code_metadata.json`, `metrics.json`).

## 5. Troubleshooting
- If the dataset exceeds ~50k rows, the CLI warns about runtime cost and defaults to the full dataset when run non-interactively.
- Predictions rely on `data/reference/isco08_reference.csv`; ensure the file remains in sync with your label set.
- For low-confidence outputs (< `model.confidence_threshold`), expect the `is_fallback` flag to be `true`; adjust the threshold in `config.yaml` to tune behaviour.

## 6. Configuration Reference
- **Data block**
  - `raw_dir`: where the unprocessed CSVs live (input to preprocessing).
  - `processed_dir`: destination for generated `train.csv`, `val.csv`, and `test.csv` splits.
  - `reference_file`: canonical ISCO lookup used for validation and human-readable labels.
  - `skip_clustering`, `skip_clustering_threshold`: disable embedding clustering entirely or only when the dataset exceeds the threshold.
- **Model block**
  - `max_seq_length`: token limit for RoBERTa; the service truncates any longer inputs.
  - `confidence_threshold`: probability cut-off controlling when a prediction is accepted versus marked as fallback.
  - `four_digit_only`: restrict predictions to 4-digit ISCO codes; set `false` to allow broader code levels.
- **Training block**
  - `gradient_accumulation_steps`: effective batch multiplier (increase for limited GPU/VRAM).
  - `warmup_steps`, `weight_decay`, `learning_rate`: standard optimisation knobs; adjust carefully and log changes.
  - `early_stopping_patience`: number of evaluation rounds without improvement before training stops.
  - `force_update_best` (optional): when true, promotes the latest model even if metrics regress.
- **Output block**
  - `model_dir`: parent directory for timestamped checkpoints after each run.
  - `best_model_dir`: canonical location used by inference; do not delete unless resetting the service.

## 7. Training & Evaluation Details
- During training the pipeline logs metrics every `logging_steps` updates and evaluates according to `evaluation_strategy` (default per epoch).
- Early stopping is active when `early_stopping_patience > 0`; the trainer tracks `eval_loss` and halts once the metric plateaus.
- After training completes, the helper compares the best `eval_loss` against `models/best_model/metrics.json` and only replaces the promoted model if the new run is better (unless forced via CLI or config).
- Evaluation (`python main.py --skip-training`) loads the promoted model and reports accuracy, 3-digit accuracy, macro F1, and top-3 accuracy in both the console and the timestamped log file.
- Check the latest log in `logs/` for full metric history, runtime durations, and any warnings about class imbalance or missing labels.
