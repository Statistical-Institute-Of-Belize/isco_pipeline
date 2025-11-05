# Repository Guidelines

## Project Structure & Module Organization
Core training workflow is orchestrated by `main.py`, which chains preprocessing, training, prediction, and evaluation. Reusable modules live in `src/` (`preprocess.py`, `model.py`, `predict.py`, `evaluate.py`, `utils.py`) and assume configuration from `config.yaml`. Data artefacts are segregated under `data/` (`raw/`, `processed/`, `corrections/`, `reference/`, `review/`); keep large intermediate files out of version control. Trained checkpoints and label metadata persist in `models/`, while runtime logs stream into `logs/` and timestamped files created at launch. The FastAPI service resides in `api/` with routers under `api/routers/` and an entry point in `api_server.py` for local serving.

## Build, Test, and Development Commands
1. `python -m venv venv && source venv/bin/activate` — create an isolated environment (skip if reusing the repo’s `venv/`).
2. `pip install -r requirements.txt` — install both pipeline and API dependencies.
3. `python main.py --config config.yaml` — run the full pipeline: preprocess, train, evaluate, and log metrics.
4. `python main.py --skip-training --input data/review/to_review_20250305.csv --explain` — reuse the latest trained model for batch predictions with optional SHAP explanations.
5. `python api_server.py` — start the development API on `http://localhost:8000`; Swagger lives at `/docs`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, `snake_case` functions, and descriptive module-level docstrings as seen across `src/`. Classes (for example, `MPSOptimizedTrainer`) use `PascalCase`. Log via the shared `logging` configuration instead of ad-hoc prints so output lands in both console and `logs/`. YAML keys stay lowercase with underscores; keep secrets and large paths in `config.yaml`, not hard-coded.

## Testing Guidelines
Automated tests are not yet present. Prefer `pytest`-style modules under a new `tests/` directory named `test_<feature>.py`. Mock large models and use lightweight CSV fixtures in `data/review/` to keep runs fast. For end-to-end validation, rerun `python main.py --skip-training` to trigger evaluation against the processed splits, and capture the reported accuracy/F1 metrics in PR notes.

## Commit & Pull Request Guidelines
Existing history uses short, lowercase subjects (`cleanup`, `docker setup`); continue with present-tense imperatives such as `add batch predictor`. Reference issues in the footer when applicable. Pull requests should outline intent, summarize pipeline or API impacts, note data dependencies, and include screenshots or sample JSON for API changes. Request review from a maintainer before merging.
