# ISCO Pipeline

A cohesive training and inference pipeline for predicting ISCO-08 occupation codes from job titles and duty descriptions. The project bundles a command-line workflow for data preparation, model training, evaluation, and batch prediction, plus a FastAPI service for online scoring.

[Full Docs](https://statistical-institute-of-belize.github.io/isco_pipeline/)

## Quick Start
- **Create env & install**
  ```bash
  python -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  ```
- **Prepare data**: place your CSV under `data/raw/` (default name `historical_records.csv`) with `job_title`, `duties_description`, and `isco_code` columns.
- **Run full pipeline**
  ```bash
  python main.py --config config.yaml
  ```
  This runs preprocessing, trains a RoBERTa classifier, evaluates it, and writes artifacts to `data/processed/` and `models/`.
- **Quick prediction-only run** (use last promoted model)
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

## Configuration
Key settings live in `config.yaml` and are grouped by concern:
- **`data`** — file locations and preprocessing switches.
  - `raw_dir`, `processed_dir`, `corrections_dir`, `reference_dir`: directory roots for the pipeline stages.
  - `review_dir`: optional drop zone for exporting prediction review sets.
  - `validate_against_reference`: check predicted codes against the official lookup at `reference_file`.
  - `skip_clustering`, `skip_clustering_threshold`: control the optional embedding clustering used during preprocessing.
- **`model`** — architecture and inference behaviour.
  - `name`: Hugging Face model identifier (default `roberta-base`).
  - `max_seq_length`: maximum token length applied consistently across training and inference.
  - `batch_size`, `epochs`, `learning_rate`: training hyperparameters.
  - `confidence_threshold`: floor for accepting predictions; lower values yield more automatic predictions, higher values mark more rows as fallbacks.
  - `four_digit_only`: constrain labels to 4-digit ISCO codes.
- **`training`** — runtime controls for the Hugging Face trainer.
  - `enable_optimizations`: toggles gradient checkpointing/mixed precision (CUDA only).
  - `gradient_accumulation_steps`: multiplies effective batch size without increasing memory.
  - `evaluation_strategy`, `logging_steps`: frequency of validation and logging updates.
  - `early_stopping_patience`: number of evaluation rounds without improvement before halting; enables automatic best-model promotion.
  - `save_strategy`, `save_total_limit`: checkpoint cadence and retention.
  - `mps_*` keys: Apple Silicon-specific mixed-precision tuning.
- **`output`** — destinations for checkpoints (`model_dir`) and the promoted model (`best_model_dir`).

Adjust paths to match your filesystem before first run. Override individual values on the CLI by editing the YAML or programmatically modifying the config dict before passing it into the pipeline.

## Repository Tour
- `main.py` — CLI entry point orchestrating preprocess → train → predict → evaluate.
- `src/` — reusable modules (preprocessing, modeling, prediction, evaluation, utilities).
- `api/` — FastAPI service, routers, and dependency wiring.
- `models/` — timestamped checkpoints and the latest promoted model in `best_model/`.
- `docs/` — supplementary guides for developers and operators.

See `docs/` for deeper developer and user guidance, including troubleshooting tips and contribution practices.
