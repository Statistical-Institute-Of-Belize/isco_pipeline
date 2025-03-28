# ISCO Classification Pipeline: User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
   - [Training](#training)
   - [Prediction/Inference](#predictioninference)
   - [Fine-tuning](#fine-tuning)
   - [Evaluation](#evaluation)
6. [Understanding Outputs](#understanding-outputs)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Introduction

The ISCO Classification Pipeline is a machine learning system designed to classify occupational data (job titles and duties descriptions) into standardized 4-digit ISCO-08 codes. When the system has low confidence in a prediction, it falls back to 3-digit codes. The pipeline leverages RoBERTa, a state-of-the-art language model, and is optimized for the MacBook Pro M3 Pro with 18GB RAM.

### Key Features

- Text preprocessing with cleaning and outlier detection
- Automated clustering to identify inconsistent labels
- RoBERTa-based classification with confidence-based fallback mechanism
- Support for fine-tuning with manual corrections
- Detailed evaluation metrics and error analysis
- SHAP explanations for predictions
- Memory-efficient processing of large datasets in chunks

## Installation

### System Requirements

- Python 3.8 or later
- macOS with M3 Pro chip (or equivalent) recommended
- 18GB RAM or more
- At least 10GB free disk space

### Setup Instructions

1. Clone the repository (if applicable) or navigate to the project directory

2. Create and activate a virtual environment:
   ```bash
   cd isco_pipeline
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   python -c "import torch; print(f'PyTorch installed: {torch.__version__}')"
   python -c "import transformers; print(f'Transformers installed: {transformers.__version__}')"
   ```

## Data Preparation

### Required Data Format

All input data must be in CSV format with specific columns:

#### Training Data

Place your training data in `isco_pipeline/data/raw/historical_records.csv` with these columns:
- `job_title`: The title of the occupation
- `duties_description`: Description of job duties and responsibilities
- `isco_code`: 4-digit ISCO-08 code for the occupation

Example:
```csv
job_title,duties_description,isco_code
Math Teacher,"Teach mathematics to high school students, prepare lessons, grade assignments",2330
Software Developer,"Develop applications using Python, debug code, write documentation",2512
```

#### Prediction Data

For prediction on new data, your CSV should include:
- `job_title`: The title of the occupation
- `duties_description`: Description of job duties and responsibilities

Example:
```csv
job_title,duties_description
Data Scientist,"Analyze large datasets, create machine learning models, present findings"
HR Manager,"Oversee recruitment, manage employee relations, implement HR policies"
```

#### Correction Data

For fine-tuning with corrections, create CSVs in `isco_pipeline/data/corrections/` with:
- `text`: Combined job title and description (in the format "title | description")
- `corrected_isco_code`: The correct 4-digit ISCO-08 code

Example:
```csv
text,corrected_isco_code
laboratory assistant | perform routine laboratory tests and experiments,3212
marketing coordinator | plan and execute marketing campaigns,2431
```

### Directory Structure

Ensure the following directories exist for data storage:
```
isco_pipeline/data/
├── raw/                 # Place training data here
├── processed/           # Preprocessed data (created automatically)
├── corrections/         # Place correction files here
├── reference/           # ISCO-08 reference files (optional)
└── review/              # Low-confidence predictions (created automatically)
```

## Configuration

The pipeline's behavior is controlled by the `config.yaml` file. Here are the key configuration options:

### Model Configuration

```yaml
model:
  name: "roberta-base"       # Base model to use
  max_seq_length: 128        # Maximum sequence length
  batch_size: 16             # Batch size for training
  epochs: 3                  # Number of training epochs
  learning_rate: 2.0e-5      # Learning rate
  confidence_threshold: 0.9  # Threshold for 4-digit vs 3-digit codes
  four_digit_only: true      # Only use 4-digit ISCO codes (improved precision)
```

### Training Configuration

```yaml
training:
  enable_optimizations: true  # Enable mixed precision and gradient checkpointing
  mps_mixed_precision: true   # Enable mixed precision for Apple Silicon
  mps_memory_efficient: true  # Use memory-efficient mode for Apple Silicon
  gradient_accumulation_steps: 4  # Accumulate gradients for larger effective batch size
  early_stopping_patience: 2  # Stop training if no improvement after N epochs
```

### Directory Configuration

```yaml
data:
  raw_dir: "data/raw/"            # Raw data location
  processed_dir: "data/processed/" # Processed data location
  corrections_dir: "data/corrections/" # Corrections location
  reference_file: "data/reference/isco08_reference.csv" # ISCO-08 reference file path
  review_dir: "data/review/"       # Low-confidence data location
  validate_against_reference: true # Validate ISCO codes against reference file
  skip_clustering: false          # Skip clustering (faster for large datasets)
  skip_clustering_threshold: 100000 # Skip clustering for datasets larger than this
output:
  model_dir: "models/"            # Model outputs location
  best_model_dir: "models/best_model/" # Best model location
```

### Explainability Configuration

```yaml
explainability:
  sample_size: 10  # Number of explanations to generate
```

## Running the Pipeline

### Training

To train the model from scratch:

```bash
python main.py [OPTIONS]
```

Options for training:
- `--config CONFIG_PATH`: Path to config file (default: `config.yaml`)
- `--enable-optimizations`: Enable mixed precision training and gradient checkpointing

Example:
```bash
python main.py --config custom_config.yaml --enable-optimizations
```

### Prediction/Inference

To predict ISCO codes for new data:

```bash
python main.py --skip-training --input INPUT_PATH [OPTIONS]
```

Options for prediction:
- `--skip-training`: Skip training and use the existing model
- `--input INPUT_PATH`: Path to the input CSV file
- `--explain`: Generate SHAP explanations for predictions

Example:
```bash
python main.py --skip-training --input data/new_jobs.csv --explain
```

### Fine-tuning

To fine-tune the model with manual corrections:

```bash
python main.py --skip-training --fine-tune [OPTIONS]
```

Options for fine-tuning:
- `--skip-training`: Skip initial training (use existing model)
- `--fine-tune`: Enable fine-tuning with corrections
- `--corrections-dir DIR_PATH`: Directory with correction CSVs (default: `data/corrections/`)

Example:
```bash
python main.py --skip-training --fine-tune --corrections-dir data/my_corrections/
```

### Evaluation

Evaluation is run automatically after training, fine-tuning, or prediction. To run just the evaluation:

```bash
python main.py --skip-training
```

## Understanding Outputs

### Model Outputs

The pipeline generates the following outputs:

#### Trained Model

The best model is saved to `models/best_model/` including:
- Model weights and configuration
- Tokenizer files
- Label mappings (`label2id.json` and `id2label.json`)

#### Predictions

Predictions are saved to:
- `data/processed/predictions_YYYYMMDD.csv`: All predictions with confidence scores
- `data/review/to_review_YYYYMMDD.csv`: Low-confidence predictions that need review

Example prediction output:
```csv
job_title,duties_description,text,predicted_isco_code,confidence,is_fallback
Systems Analyst,"Design and implement computer systems",systems analyst | design and implement computer systems,2511,0.94,False
Marketing Assistant,"Assist with campaigns",marketing assistant | assist with campaigns,243,0.78,True
```

#### Explanations

If `--explain` is used:
- SHAP explanations are saved to `data/processed/explanations/explanation_X.html`
- These HTML files show which words influenced the prediction

#### Evaluation Metrics

Evaluation results are saved to:
- `models/runs/run_YYYYMMDD/metrics.json`: Overall metrics
- `models/runs/run_YYYYMMDD/error_analysis/`: Detailed error analysis

Key metrics include:
- Accuracy
- Macro F1-score
- Top-3 accuracy
- 3-digit accuracy
- Error rates by major group

### Logs

The pipeline logs progress information to the console, including:
- Preprocessing status
- Training progress
- Evaluation metrics
- Error information

## Troubleshooting

### Common Issues

#### Out of Memory Errors

**Symptoms**: Process killed during embedding generation or training
**Solutions**:
- Reduce `batch_size` in `config.yaml`
- Ensure `enable_optimizations` is set to `true`
- Increase `gradient_accumulation_steps` (allows smaller batch size)
- Enable `mps_memory_efficient` for Apple Silicon
- Process data in smaller chunks by editing `chunk_size` in `preprocess.py`
- For datasets >150K records, respond "y" when prompted to use CPU instead of GPU

#### Invalid ISCO Codes

**Symptoms**: Warnings about invalid ISCO codes
**Solutions**:
- Check `data/processed/invalid_codes.csv` for records with invalid codes
- Fix the codes and add the corrections to the `data/corrections/` directory
- Check for armed forces codes (110, 210, 310) that need leading zeros (0110, 0210, 0310)
- Verify the encoding of your reference file matches your data

#### Low Accuracy

**Symptoms**: Poor performance metrics in evaluation
**Solutions**:
- Increase `epochs` in `config.yaml`
- Check `data/processed/outliers.csv` for inconsistent labels
- Add corrections for commonly misclassified examples
- Adjust `confidence_threshold` based on precision/recall tradeoff
- Try setting `four_digit_only: true` for improved precision on 4-digit codes
- Check if your dataset has rare codes causing stratification issues

### Error Logs

If you encounter errors, check:
- Terminal output for error messages
- `data/processed/outliers.csv` for data quality issues
- `models/runs/run_YYYYMMDD/error_analysis/` for patterns in errors

## Advanced Usage

### Custom Configurations

You can create custom configuration files for different scenarios:
- `config_full.yaml` for full training
- `config_quick.yaml` for rapid prototyping
- `config_production.yaml` for final deployment

Example custom config:
```yaml
model:
  name: "roberta-base"
  max_seq_length: 128
  batch_size: 8  # Smaller batch for less memory
  epochs: 5      # More epochs for better accuracy
  learning_rate: 1.0e-5  # Lower learning rate
  confidence_threshold: 0.85  # Lower threshold for more 4-digit codes
  four_digit_only: true  # Only use 4-digit ISCO codes

training:
  enable_optimizations: true
  mps_mixed_precision: true
  mps_memory_efficient: true
  gradient_accumulation_steps: 8  # Larger accumulation for small batch sizes
  early_stopping_patience: 3  # More patient early stopping
```

### Working with Large Datasets

For datasets larger than 150,000 records:
1. Split your data into multiple CSV files
2. Process each file separately
3. Combine the processed outputs for training

### Integrating with Existing Systems

The pipeline can be integrated with existing systems by:
1. Setting up a shared data directory for input/output
2. Creating scheduled jobs to process new data
3. Using the `--skip-training` option for inference on new data
4. Implementing a feedback loop using the corrections directory

### Performance Tuning

For optimal performance:
- Experiment with different `batch_size` values
- Try different `learning_rate` values
- Adjust `confidence_threshold` based on your precision/recall needs
- Use `--enable-optimizations` for faster training
- Configure `max_seq_length` based on your data characteristics

---

For additional support or custom modifications, please refer to the README or contact the development team.