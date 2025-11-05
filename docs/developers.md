# Developer Guide

## 1. Architecture Snapshot
- `main.py` orchestrates preprocessing → training → prediction → evaluation. Each stage delegates to `src/` modules.
- Preprocessing (`src/preprocess.py`) normalises text via `clean_and_combine_text`; reuse this helper for any new ingestion path.
- Training (`src/model.py`) uses Hugging Face `Trainer`/`TrainingArguments` with optional MPS optimisations and promotes the best model by comparing `eval_loss` against `models/best_model/metrics.json`.
- Inference (`src/predict.py` and `api/prediction_service.py`) loads the promoted model, applies a keyed LRU cache, and respects `model.max_seq_length` and `model.confidence_threshold` from `config.yaml`.

## 2. Local Development Workflow
1. Sync dependencies: `pip install -r requirements.txt` (use editable installs for local packages if needed).
2. Use `python main.py --config config.yaml --skip-training` to validate preprocessing/evaluation changes without retraining.
3. For API changes, run `python api_server.py` and exercise endpoints using the Swagger UI or `curl`.
4. Keep large intermediate data out of git; working files should live under `data/` or `tmp_trainer/`.

## 3. Coding Practices
- Stick to PEP 8 with 4-space indentation; prefer descriptive names and module-level constants over magic numbers.
- Route all logging through the shared logger; avoid print statements so output lands in both console and rotating log files.
- Avoid duplicating configuration: read from `config.yaml` via helpers in `src/utils.py` and propagate values through function parameters instead of globals.
- When introducing new preprocessing or prediction logic, ensure the CLI and API paths both call the same helper to keep behaviour aligned.

## 4. Testing & Verification
- Add `pytest` suites under `tests/`; mock heavy transformers components to keep runs quick.
- Prioritise regression coverage for: text formatting helpers, prediction caching, attention-based explanation generation, and API batch handlers.
- Before opening a PR, run targeted scripts (`python -m py_compile <module>` or lightweight smoke runs) and note the results in the PR description. Full end-to-end training is optional unless core training code changed.

## 5. Known Gaps / TODOs
- Replace interactive `input()` prompts in `src/preprocess.py` and `src/model.py` with config-driven behaviour for non-TTY environments.
- Revisit the very low default `model.confidence_threshold` once business requirements are clarified.
- Add automated tests for the attention-based explanation pipeline and the API CSV upload route.

## 6. Configuration Cheat Sheet
- Access configuration via `src.utils.load_config`; keep overrides explicit to aid reproducibility.
- Common keys and their downstream consumers:
  - `data` ➜ preprocessing, evaluation, and API reference lookups. Ensure relative paths resolve from the repository root.
  - `model.name` ➜ passed to Hugging Face initialisation; updating requires compatible tokenizers and possibly retraining checkpoints.
  - `model.max_seq_length` ➜ shared between `prepare_datasets` and API tokenisation. Changes necessitate reprocessing training data.
  - `training.enable_optimizations` ➜ toggles gradient checkpointing and fp16 (CUDA only). On Apple Silicon only the MPS-specific toggles apply.
  - `training.early_stopping_patience` ➜ automatically enables `load_best_model_at_end` in `TrainingArguments`.
  - `output.best_model_dir` ➜ single source of truth for inference; do not point the API to a different location without updating the YAML.

## 7. Training & Evaluation Flow
- `train_model` constructs `TrainingArguments` using the config values, applies Apple Silicon optimisations when available, and records elapsed time for observability.
- Post-training, `save_best_model` promotes the run if `eval_loss` improved; metrics and mappings are written alongside the weights for traceability.
- CLI evaluation (`evaluate_model(config)`) reloads processed validation/test splits, reports aggregate metrics, and can be invoked independently by running `python main.py --skip-training`.
- When experimenting with hyperparameters, favour timestamped checkpoints in `models/` for forensic analysis; only copy artifacts into `best_model_dir` once they outperform the incumbent.
