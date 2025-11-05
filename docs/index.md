# ISCO Pipeline

A cohesive training and inference pipeline for predicting ISCO-08 occupation codes from job titles and duty descriptions. The project bundles a command-line workflow for data preparation, model training, evaluation, and batch prediction, plus a FastAPI service for online scoring.

## Quick Start

### Create Environment & Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Prepare Data

Place your CSV under `data/raw/` (default name `historical_records.csv`) with `job_title`, `duties_description`, and `isco_code` columns.

### Run Full Pipeline

```bash
python main.py --config config.yaml
```

This runs preprocessing, trains a RoBERTa classifier, evaluates it, and writes artifacts to `data/processed/` and `models/`.

### Quick Prediction-Only Run

Use the last promoted model:

```bash
python main.py --skip-training --input data/review/sample_jobs.csv --explain
```

Adds predictions and optional attention-based explanations to the input CSV.

## API Usage

1. Ensure the best model exists at `models/best_model/` (training updates this automatically when metrics improve).

2. Start the API:
   ```bash
   python api_server.py
   ```
   The server listens on `http://localhost:8000` with interactive docs at `/docs`.

3. Example single prediction request:
   ```bash
   curl -X POST http://localhost:8000/predict/job \
     -H "Content-Type: application/json" \
     -d '{"job_title":"Software Engineer","duties_description":"Design and build backend services"}'
   ```

## Repository Tour

- **`main.py`** — CLI entry point orchestrating preprocess → train → predict → evaluate.
- **`src/`** — reusable modules (preprocessing, modeling, prediction, evaluation, utilities).
- **`api/`** — FastAPI service, routers, and dependency wiring.
- **`models/`** — timestamped checkpoints and the latest promoted model in `best_model/`.
- **`docs/`** — supplementary guides for developers and operators.

## Next Steps

- [User Guide](users.md) — detailed walk-through for running training, evaluation, and predictions
- [Developer Guide](developers.md) — architectural notes and development workflow
- [Configuration Reference](configuration.md) — comprehensive configuration options
- [API Reference](api.md) — API endpoints and usage examples
