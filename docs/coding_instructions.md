# Optimized Detailed Plan Outline for ISCO Classification Pipeline (Pseudo-Code with Instructions)
# This plan provides pseudo-code and detailed comments to guide a code-specialized LLM in implementing
# the ISCO classification pipeline. Instructions focus on WHAT to do, with HOW specified only for critical details.

# 1. Project Setup
# Objective: Set up the directory structure and install dependencies.

# Pseudo-Code with Instructions
// Step 1: Create directory structure
// Instruction: Create the following directories and empty files:
// - isco_pipeline/data/{raw,processed,corrections,reference,review}
// - isco_pipeline/models/{best_model,runs}
// - isco_pipeline/src
// - Files: config.yaml, main.py, requirements.txt, README.md, src/{__init__.py,preprocess.py,model.py,predict.py,evaluate.py,utils.py}
create_project_structure()

// Step 2: Write requirements.txt
// Instruction: Write a requirements file with these libraries and versions:
// torch==2.0.1, transformers==4.35.0, pandas==2.0.3, scikit-learn==1.3.0, pyyaml==6.0.1,
// shap==0.43.0, seaborn==0.12.2, matplotlib==3.7.2, numpy==1.24.3
write_requirements_file("isco_pipeline/requirements.txt")

// Step 3: Write config.yaml
// Instruction: Write a YAML config file with:
// data: {raw_dir: "data/raw/", processed_dir: "data/processed/", corrections_dir: "data/corrections/",
//        reference_dir: "data/reference/", review_dir: "data/review/"}
// model: {name: "roberta-base", max_seq_length: 128, batch_size: 16, epochs: 3, learning_rate: 2.0e-5,
//         confidence_threshold: 0.9}
// training: {enable_optimizations: true}
// output: {model_dir: "models/", best_model_dir: "models/best_model/"}
// explainability: {sample_size: 10}
write_config_file("isco_pipeline/config.yaml")

# 2. Data Preprocessing (`src/preprocess.py`)
# Objective: Preprocess raw CSV data, cluster for outliers, and split into train/validation/test sets.

# Pseudo-Code with Instructions
// Import: Libraries for data processing, clustering, tokenization, and YAML handling

function load_config(config_path="config.yaml")
    // Instruction: Load and return the YAML config file as a dictionary
    return config

function cluster_embeddings(texts, model, tokenizer, chunk_size=10000)
    // Instruction: Cluster texts using embeddings from model and tokenizer to identify outliers;
    // Process in chunks of 10000 to optimize memory; return cluster labels
    return clustering_labels

function preprocess_data(input_csv, config)
    // Instruction: Load CSV with columns "job_title", "duties_description", "isco_code"; raise error if missing
    df = load_csv(input_csv)

    // Instruction: Fill missing "duties_description" with "unknown"; combine into "text" column as
    // "job_title | duties_description"; clean by lowercasing and removing non-word characters except "|"
    df["text"] = clean_and_combine_text(df["job_title"], df["duties_description"])

    // Instruction: Ensure "isco_code" is 4-digit string; save invalid rows to config["data"]["processed_dir"]/invalid_codes.csv
    df = validate_isco_codes(df, config["data"]["processed_dir"])

    // Instruction: Cluster df["text"] using embeddings; save clusters with >1 unique ISCO code to
    // config["data"]["processed_dir"]/outliers.csv
    df["cluster"] = cluster_texts(df["text"], config["model"]["name"])
    save_outliers(df, config["data"]["processed_dir"])

    // Instruction: Split df into train (70%), val (15%), test (15%) with stratification by "isco_code";
    // Save each split as CSV with "text" and "isco_code" to config["data"]["processed_dir"]
    split_and_save_data(df, config["data"]["processed_dir"])

if __name__ == "__main__":
    // Instruction: Load config; ensure config["data"]["processed_dir"] exists; preprocess
    // config["data"]["raw_dir"]/historical_records.csv
    run_preprocessing()

# 3. Model Training (`src/model.py`)
# Objective: Train RoBERTa on preprocessed data and save the best model.

# Pseudo-Code with Instructions
// Import: Libraries for data handling, RoBERTa model, training, and JSON saving

function load_config(config_path="config.yaml")
    // Instruction: Load and return the YAML config file as a dictionary
    return config

class ISCODataset
    // Instruction: Define a dataset class to hold tokenized texts and labels for training
    define_dataset_class()

function train_model(config)
    // Instruction: Load train.csv and val.csv; create label2id mapping from unique "isco_code"
    train_df, val_df, label2id = load_and_map_labels(config["data"]["processed_dir"])

    // Instruction: Prepare train and val datasets from text and labels using config["model"]["max_seq_length"]
    train_dataset, val_dataset = prepare_datasets(train_df, val_df, label2id, config["model"]["max_seq_length"])

    // Instruction: Initialize RoBERTa model with config["model"]["name"] and num_labels=len(label2id);
    // Optimize for M3 Pro with appropriate device
    model = initialize_model(config["model"]["name"], len(label2id))

    // Instruction: Train model with: batch_size=16, epochs=3, learning_rate=2.0e-5, mixed precision enabled,
    // save and evaluate each epoch, keep best model based on eval loss
    trainer = train_model_with_params(model, config)

    // Instruction: Save best model to config["output"]["best_model_dir"]; save label2id and id2label as JSON
    save_best_model(trainer, config["output"]["best_model_dir"], label2id)

if __name__ == "__main__":
    // Instruction: Load config; train model
    run_training()

# 4. Prediction (`src/predict.py`)
# Objective: Predict ISCO codes, flag low-confidence cases, and generate explanations.

# Pseudo-Code with Instructions
// Import: Libraries for data handling, RoBERTa prediction, SHAP explainability, and caching

function load_config(config_path="config.yaml")
    // Instruction: Load and return the YAML config file as a dictionary
    return config

function predict_single(text, model, tokenizer, label_map, threshold) with caching
    // Instruction: Predict ISCO code for text; return 4-digit code if confidence >= 0.9, else 3-digit;
    // Return dict with text, isco_code, confidence, is_fallback; cache results for efficiency
    return prediction_dict

function predict_batch(texts, model, tokenizer, label_map, config, explain=False)
    // Instruction: Predict ISCO codes for all texts; save results to config["data"]["processed_dir"]/predictions.csv
    predictions = predict_all(texts)

    // Instruction: Flag predictions with confidence < 0.9; save to config["data"]["review_dir"]/to_review_<YYYYMMDD>.csv
    flag_low_confidence(predictions, config["data"]["review_dir"])

    // Instruction: If explain=True, generate explanations for first 10 texts; save to
    // config["data"]["processed_dir"]/explanation_<index>.html
    if explain then generate_explanations(texts, config["explainability"]["sample_size"])

if __name__ == "__main__":
    // Instruction: Load config, model, tokenizer, and label_map; predict on test.csv with explanations
    run_prediction()

# 5. Evaluation (`src/evaluate.py`)
# Objective: Evaluate model performance with metrics and error analysis.

# Pseudo-Code with Instructions
// Import: Libraries for data handling, RoBERTa evaluation, metrics, and visualization

function load_config(config_path="config.yaml")
    // Instruction: Load and return the YAML config file as a dictionary
    return config

class ISCODataset
    // Instruction: Define a dataset class to hold tokenized texts and labels for evaluation
    define_dataset_class()

function evaluate_model(config)
    // Instruction: Load test.csv and create dataset with label mapping
    test_df, label2id = load_test_data(config)
    test_dataset = prepare_dataset(test_df, label2id, config["model"]["max_seq_length"])

    // Instruction: Evaluate model on test_dataset; compute macro F1-score, accuracy, and top-3 accuracy
    metrics = compute_metrics(test_dataset)

    // Instruction: Perform error analysis: top 10 misclassifications, error rates by major group,
    // confusion matrix heatmap; save to config["output"]["model_dir"]/runs/run_<YYYYMMDD>/error_analysis/
    perform_error_analysis(metrics, test_df, config["output"]["model_dir"])

if __name__ == "__main__":
    // Instruction: Load config; evaluate model
    run_evaluation()

# 6. CLI Interface (`main.py`)
# Objective: Provide a command-line interface to run the pipeline.

# Pseudo-Code with Instructions
// Import: Libraries for argument parsing and pipeline components

function load_config(config_path)
    // Instruction: Load and return the YAML config file as a dictionary
    return config

function main()
    // Instruction: Parse arguments: --config (default="config.yaml"), --skip-training, --fine-tune,
    // --corrections-dir (default="data/corrections/"), --enable-optimizations, --explain, --input
    args = parse_arguments()

    // Instruction: Load config; update optimizations if flag set
    config = load_and_update_config(args)

    if not args.skip_training
        // Instruction: Preprocess and train model
        run_preprocessing_and_training(config)

    if args.fine_tune
        // Instruction: Load corrections from args.corrections_dir; fine-tune model with 1-2 epochs,
        // learning_rate=1.0e-5; save updated model
        run_fine_tuning(config, args.corrections_dir)

    if args.input
        // Instruction: Predict on input CSV with optional explanations
        run_prediction(args.input, config, args.explain)

    // Instruction: Evaluate model
    run_evaluation(config)

if __name__ == "__main__":
    main()

# 7. Execution Sequence
# Instructions
// Instruction: To Run full pipeline: python isco_pipeline/main.py --config isco_pipeline/config.yaml
// To Run prediction only: python isco_pipeline/main.py --skip-training --input isco_pipeline/data/new_data.csv --explain
