
---

### Downloadable Text Artifact: `isco_full_detailed_plan.txt`

```
# Updated Detailed Plan Outline for ISCO Classification Pipeline
# This plan outlines a Python-based machine learning pipeline to classify occupational data into
# 4-digit ISCO-08 codes, with enhancements for accuracy, efficiency, and maintainability.
# Current date: March 05, 2025.

# 1. Project Overview
## Objective
Develop a Python-based machine learning pipeline to classify occupational data (job titles and duties descriptions)
into 4-digit ISCO-08 codes, with a fallback to 3-digit codes when confidence is low. The pipeline prioritizes accuracy,
efficiency, and maintainability, supports fine-tuning with manual corrections, and includes advanced features for
error analysis, active learning, data cleaning, and explainability.

## Key Constraints
- Hardware: MacBook Pro, M3 Pro processor, 18GB RAM.
- Data: 150,000 historical records (CSV format) with potentially noisy labels.
- Model: RoBERTa for text classification.
- Standards: ISCO-08 (436 4-digit codes, ~130 3-digit codes).

# 2. Project Structure
```
isco_pipeline/
├── data/                    # Input/output data
│   ├── raw/                # Raw CSV files (e.g., historical_records.csv)
│   ├── processed/          # Preprocessed data (e.g., train.csv, val.csv, test.csv)
│   ├── corrections/        # Manual corrections (e.g., corrections_2025.csv)
│   ├── reference/          # ISCO-08 manual data (e.g., isco08_definitions.csv)
│   └── review/             # Records flagged for human review (e.g., to_review_20250305.csv)
├── models/                 # Saved models and metadata
│   ├── best_model/         # Best-performing model checkpoint
│   └── runs/               # Model run logs (e.g., run_20250305/)
├── src/                    # Source code
│   ├── __init__.py
│   ├── preprocess.py       # Text cleaning, clustering, and data prep
│   ├── model.py            # RoBERTa model definition and training
│   ├── predict.py          # Inference logic with active learning and explainability
│   ├── evaluate.py         # Evaluation metrics and error analysis
│   └── utils.py            # Helper functions (e.g., caching, logging)
├── config.yaml             # Configuration file
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

### Rationale
- Separation of Concerns: Data, models, and code are cleanly separated.
- Scalability: Easy to add new modules or data types.
- Maintainability: Simple structure with clear responsibilities per file.

# 3. Pipeline Components

## 3.1 Data Preprocessing
### Specifications
- Input: CSV files with columns: `job_title`, `duties_description`, `isco_code` (4-digit).
- Output: Processed CSV files (`train.csv`, `val.csv`, `test.csv`) with columns: `text` (combined title + duties), `label` (ISCO code).
- Steps:
  1. Text Cleaning:
     - Lowercase text.
     - Remove punctuation, extra whitespace, and special characters (except separators).
     - Handle missing values (e.g., replace NaN with "unknown").
  2. Text Joining:
     - Combine `job_title` and `duties_description` with a separator (e.g., `|`).
     - Example: "Teacher | Plan lessons, teach students, grade assignments".
  3. Label Validation:
     - Ensure `isco_code` is a valid 4-digit ISCO-08 code (e.g., 2310).
     - Flag invalid codes for manual review.
  4. Embedding-Based Clustering:
     - Extract RoBERTa embeddings for all records.
     - Cluster similar texts using DBSCAN to identify outliers or inconsistent labels.
     - Flag potential errors (e.g., same cluster with different codes) to `data/processed/outliers.csv`.
  5. Train-Validation-Test Split:
     - 70% train, 15% validation, 15% test.
     - Stratify by ISCO code to preserve distribution.
- Optimization: Process in chunks (e.g., 10,000 records) to fit 18GB RAM.

### Instructions
- Implement in `src/preprocess.py`:
```python
from transformers import RobertaModel, RobertaTokenizer
from sklearn.cluster import DBSCAN

def cluster_embeddings(texts, model, tokenizer, chunk_size=10000):
    embeddings = []
    for i in range(0, len(texts), chunk_size):
        batch = tokenizer(texts[i:i+chunk_size], return_tensors="pt", padding=True, truncation=True)
        embeddings.append(model(**batch).last_hidden_state.mean(dim=1).detach().cpu().numpy())
    embeddings = np.vstack(embeddings)
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(embeddings)
    return clustering.labels_

def preprocess_data(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    df["text"] = df["job_title"] + " | " + df["duties_description"].fillna("unknown")
    df["text"] = df["text"].str.lower().str.replace(r"[^\w\s|]", "", regex=True)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    df["cluster"] = cluster_embeddings(df["text"].tolist(), model, tokenizer)
    outliers = df.groupby("cluster").filter(lambda x: x["isco_code"].nunique() > 1)
    outliers.to_csv(f"{output_dir}/outliers.csv")
    train, val, test = train_test_split(df, test_size=0.3, stratify=df["isco_code"])
    val, test = train_test_split(val, test_size=0.5, stratify=val["isco_code"])
    train.to_csv(f"{output_dir}/train.csv")
    val.to_csv(f"{output_dir}/val.csv")
    test.to_csv(f"{output_dir}/test.csv")
```

## 3.2 Model Definition and Training
### Specifications
- Model: RoBERTa (`roberta-base`, ~125M parameters) from Hugging Face.
- Task: Multi-class classification (436 classes for 4-digit codes).
- Input: Tokenized `text` (max length: 128 tokens).
- Output: Predicted 4-digit ISCO code + confidence score.
- Training Parameters:
  - Batch size: 16 (fits 18GB RAM on M3 Pro).
  - Learning rate: 2e-5.
  - Epochs: 3-5.
  - Optimizer: AdamW.
  - Loss: Cross-entropy.
- Optimizations:
  - Use mixed-precision training (`torch.cuda.amp` or MPS equivalent) to reduce memory usage.
  - Enable gradient checkpointing if memory is tight.
- Fallback Mechanism:
  - If confidence < 0.9, predict 3-digit code (truncate 4-digit prediction, e.g., 2310 → 231).
  - Flag 3-digit predictions in output.

### Instructions
- Implement in `src/model.py`:
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=436)

training_args = TrainingArguments(
    output_dir="../models/runs/",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    fp16=True,  # Mixed precision
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
)
```

## 3.3 Prediction
### Specifications
- Input: Preprocessed text (single record or batch).
- Output: JSON with `isco_code` (4-digit or 3-digit), `confidence`, `is_fallback` (boolean), and optional SHAP explanation.
- Active Learning: Flag low-confidence predictions (<0.9) to `data/review/to_review_<date>.csv`.
- Explainability: Generate SHAP explanations for a sample of predictions.
- Optimizations: Use LRU caching (`functools.lru_cache`) for repeated inputs; process in batches.

### Instructions
- Implement in `src/predict.py`:
```python
import shap
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_single(text, model, tokenizer, threshold=0.9):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    code_idx = probs.argmax().item()
    confidence = probs[0, code_idx].item()
    isco_code = isco_labels[code_idx] if confidence >= threshold else isco_labels[code_idx][:3]
    return {"text": text, "isco_code": isco_code, "confidence": confidence, "is_fallback": confidence < threshold}

def flag_for_review(predictions, output_path):
    low_conf = [(p["text"], p["isco_code"], p["confidence"]) for p in predictions if p["confidence"] < threshold]
    pd.DataFrame(low_conf, columns=["text", "predicted_code", "confidence"]).to_csv(output_path)

def explain_prediction(model, tokenizer, text, output_path):
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer([text])
    shap.plots.text(shap_values[0]).save(f"{output_path}/explanation_{hash(text)}.html")
```

## 3.4 Fine-Tuning with Manual Corrections
### Specifications
- Input: CSV in `data/corrections/` with columns: `text`, `corrected_isco_code`.
- Process:
  1. Load existing model from `models/best_model/`.
  2. Append corrections to training data.
  3. Fine-tune for 1-2 epochs with learning rate 1e-5.
- Output: Updated model saved to `models/best_model/`.

### Instructions
- Add to `main.py` with `--fine-tune` and `--corrections-dir` options:
```python
trainer.train()  # Fine-tune on combined original + corrections data
```

## 3.5 Model Evaluation
### Specifications
- Metrics:
  - Macro-averaged F1-score (primary).
  - Accuracy.
  - Top-3 accuracy.
  - Percentage of 3-digit fallbacks.
- Enhanced Error Analysis:
  - Common misclassifications (top 10 confused pairs).
  - Error rates by ISCO major groups (1-digit) and subcategories (2-digit, 3-digit).
  - Visualizations (e.g., confusion matrix heatmap).
- Output: Metrics to `models/runs/run_<date>/metrics.json`; error analysis to `models/runs/run_<date>/error_analysis/`.

### Instructions
- Implement in `src/evaluate.py`:
```python
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(trainer, eval_dataset, isco_labels, output_dir):
    preds = trainer.predict(eval_dataset)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    metrics = {"f1_macro": f1, "accuracy": acc}
    cm = confusion_matrix(y_true, y_pred)
    top_misclassifications = pd.DataFrame(cm).stack().sort_values(ascending=False).head(10)
    major_groups = [str(code)[0] for code in isco_labels]
    group_errors = pd.DataFrame({"true": y_true, "pred": y_pred, "group": major_groups})
    error_rates = group_errors.groupby("group").apply(lambda x: (x["true"] != x["pred"]).mean())
    pd.DataFrame(metrics, index=[0]).to_json(f"{output_dir}/metrics.json")
    top_misclassifications.to_csv(f"{output_dir}/error_analysis/misclassifications.csv")
    error_rates.to_csv(f"{output_dir}/error_analysis/error_rates_by_group.csv")
    sns.heatmap(cm, annot=False).figure.savefig(f"{output_dir}/error_analysis/confusion_matrix.png")
    return metrics
```

## 3.6 Integration with ISCO Manual
### Specifications
- Reference Data: CSV in `data/reference/` with columns: `isco_code`, `description`.
- Process: Optionally concatenate with training data as additional examples.

### Instructions
- Load in `src/preprocess.py` and experiment with inclusion.

# 4. Configuration
### Specifications
- File: `config.yaml`.
- Fields:
```yaml
data:
  raw_dir: "data/raw/"
  processed_dir: "data/processed/"
  corrections_dir: "data/corrections/"
  reference_dir: "data/reference/"
  review_dir: "data/review/"
model:
  name: "roberta-base"
  max_seq_length: 128
  batch_size: 16
  epochs: 3
  learning_rate: 2e-5
  confidence_threshold: 0.9
training:
  enable_optimizations: true
output:
  model_dir: "models/"
  best_model_dir: "models/best_model/"
explainability:
  sample_size: 10
```
- Usage: Load in `main.py` with `pyyaml`.

# 5. CLI Interface
### Specifications
- Entry Point: `main.py`.
- Options:
  - `--config`: Path to config file (default: `config.yaml`).
  - `--skip-training`: Skip training, use existing model.
  - `--fine-tune`: Fine-tune with corrections.
  - `--corrections-dir`: Directory with correction CSVs.
  - `--enable-optimizations`: Enable mixed precision and checkpointing.
  - `--explain`: Generate SHAP explanations for predictions.

### Instructions
- Use `argparse`:
```python
parser.add_argument("--explain", action="store_true", help="Generate SHAP explanations")
```

# 6. Optimization for Hardware
### Specifications
- Memory: Limit batch size to 16; use mixed precision to fit 18GB RAM.
- Processor: Leverage M3 Pro’s MPS (Metal Performance Shaders) with PyTorch.
- Batch Processing: Process data in chunks (e.g., 10,000 records at a time).

### Instructions
- Enable MPS in PyTorch:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

# 7. Running on New Data
### Specifications
- Input: CSV with `job_title` and `duties_description`.
- Command: `python main.py --skip-training --input data/new_data.csv --explain`.

### Instructions
- Save predictions to `data/processed/predictions.csv`.

# 8. Implementation Steps
1. Setup Environment:
   - Install dependencies: `pip install torch transformers pandas sklearn pyyaml shap seaborn matplotlib`.
   - Create `requirements.txt`.
2. Preprocess Data: Write `preprocess.py` and test on sample CSV.
3. Train Model: Implement `model.py` and train on full dataset.
4. Evaluate: Run `evaluate.py` on validation/test sets.
5. Predict: Test `predict.py` on a small batch.
6. Fine-Tune: Simulate corrections and test fine-tuning.
7. Finalize CLI: Integrate all components in `main.py`.
```

---
