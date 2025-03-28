# ISCO Pipeline Technical Architecture

## Overview

The ISCO Classification Pipeline is designed to classify job descriptions into appropriate ISCO-08 codes. The system consists of two main components:

1. The training pipeline for model development
2. The inference pipeline (API) for making predictions

## Components

### Core Libraries

- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Provides the RoBERTa model architecture
- **FastAPI**: API framework for serving predictions
- **pandas**: Data manipulation
- **scikit-learn**: For data preprocessing and evaluation metrics

### Model Architecture

The classification model uses a fine-tuned RoBERTa architecture (roberta-base) with:

- Maximum sequence length of 128 tokens
- Mixed precision training on supported hardware (CUDA/MPS)
- Gradient checkpointing for memory efficiency
- Early stopping to prevent overfitting

## Training Pipeline

The training pipeline consists of several stages:

### 1. Data Preprocessing
- Text cleaning and normalization
- ISCO code validation against reference data
- Outlier detection using DBSCAN clustering
- Train/validation/test splitting

### 2. Model Training
- Label mapping including all valid ISCO codes
- Dynamic batch size and gradient accumulation based on dataset size
- Mixed precision training when available
- Trainer with early stopping and evaluation

### 3. Model Management
- Only saves as "best model" if it outperforms previous models
- Preserves code mappings in multiple formats for reliability
- Saves metadata about ISCO codes, including hierarchy

## Inference Pipeline

The inference engine follows these steps:

1. Load the best model, tokenizer, and label mappings
2. Preprocess input text (clean and combine)
3. Generate predictions with confidence scores
4. Apply confidence thresholding for fallback to 3-digit codes when uncertain
5. Add alternative predictions and occupation titles from reference data
6. Return structured results

## API Architecture

The API service is built with FastAPI and provides multiple endpoints:

- `/predict/job`: Single job prediction
- `/predict/batch`: Batch prediction for multiple jobs
- `/predict/csv`: CSV file upload and processing

The API uses dependency injection to load the model only once and reuse it across requests.

## Customization

The pipeline is customizable through configuration files:

- Model parameters (learning rate, batch size, epochs)
- Training optimizations (mixed precision, gradient checkpointing)
- Confidence thresholds for fallback predictions
- Data validation settings

## Deployment Options

The system supports multiple deployment strategies:

1. **Local development**: Direct Python execution
2. **Docker containerization**: For consistent environments
3. **Production deployment**: With load balancing and HTTPS