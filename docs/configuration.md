# Configuration Reference

Key settings live in `config.yaml` and are grouped by concern.

## Data Configuration

File locations and preprocessing switches:

- **`raw_dir`**, **`processed_dir`**, **`corrections_dir`**, **`reference_dir`** — directory roots for the pipeline stages.
- **`review_dir`** — optional drop zone for exporting prediction review sets.
- **`validate_against_reference`** — check predicted codes against the official lookup at `reference_file`.
- **`skip_clustering`**, **`skip_clustering_threshold`** — control the optional embedding clustering used during preprocessing.

### Example

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  corrections_dir: "data/corrections"
  reference_dir: "data/reference"
  review_dir: "data/review"
  validate_against_reference: true
  skip_clustering: false
  skip_clustering_threshold: 50000
```

## Model Configuration

Architecture and inference behaviour:

- **`name`** — Hugging Face model identifier (default `roberta-base`).
- **`max_seq_length`** — maximum token length applied consistently across training and inference.
- **`batch_size`**, **`epochs`**, **`learning_rate`** — training hyperparameters.
- **`confidence_threshold`** — floor for accepting predictions; lower values yield more automatic predictions, higher values mark more rows as fallbacks.
- **`four_digit_only`** — constrain labels to 4-digit ISCO codes.

### Example

```yaml
model:
  name: "roberta-base"
  max_seq_length: 256
  batch_size: 16
  epochs: 3
  learning_rate: 2.0e-5
  confidence_threshold: 0.1
  four_digit_only: true
```

## Training Configuration

Runtime controls for the Hugging Face trainer:

- **`enable_optimizations`** — toggles gradient checkpointing/mixed precision (CUDA only).
- **`gradient_accumulation_steps`** — multiplies effective batch size without increasing memory.
- **`evaluation_strategy`**, **`logging_steps`** — frequency of validation and logging updates.
- **`early_stopping_patience`** — number of evaluation rounds without improvement before halting; enables automatic best-model promotion.
- **`save_strategy`**, **`save_total_limit`** — checkpoint cadence and retention.
- **`mps_*` keys** — Apple Silicon-specific mixed-precision tuning.

### Example

```yaml
training:
  enable_optimizations: true
  gradient_accumulation_steps: 2
  evaluation_strategy: "epoch"
  logging_steps: 50
  early_stopping_patience: 2
  save_strategy: "epoch"
  save_total_limit: 2
  warmup_steps: 100
  weight_decay: 0.01
```

## Output Configuration

Destinations for checkpoints and the promoted model:

- **`model_dir`** — timestamped checkpoints directory.
- **`best_model_dir`** — promoted model location (used by both CLI and API).

### Example

```yaml
output:
  model_dir: "models"
  best_model_dir: "models/best_model"
```

## Configuration Best Practices

!!! tip "Override Values"
    Adjust paths to match your filesystem before first run. Override individual values on the CLI by editing the YAML or programmatically modifying the config dict before passing it into the pipeline.

!!! warning "Path Resolution"
    Ensure relative paths resolve from the repository root. All paths in the configuration are relative to the project root directory.

!!! info "API Configuration"
    The API automatically loads `models/best_model/`. Do not point the API to a different location without updating the YAML.
