import os
import sys
import argparse
import logging
import pandas as pd
import torch
from datetime import datetime

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, ensure_dir, load_isco_reference
from src.preprocess import preprocess_data
from src.model import train_model
from src.predict import predict_batch, load_model_and_mappings
from src.evaluate import evaluate_model

# Configure logging with improved formatting
import colorama
from colorama import Fore, Style
import platform

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

# Define custom log formatter with colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors and improved formatting to log messages"""
    
    COLORS = {
        'DEBUG': Style.DIM + Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    SYMBOLS = {
        'DEBUG': '🔍',
        'INFO': '✓',
        'WARNING': '⚠️',
        'ERROR': '✗',
        'CRITICAL': '❌',
    }
    
    def __init__(self, use_symbols=True):
        # Windows terminal might have issues with emoji
        self.use_symbols = use_symbols and platform.system() != 'Windows'
        super().__init__('%(asctime)s - %(levelname)s - %(message)s')
    
    def format(self, record):
        # Save the original format
        original_format = self._style._fmt
        
        # Apply color to the level name
        levelname = record.levelname
        color = self.COLORS.get(levelname, '')
        symbol = self.SYMBOLS.get(levelname, '') + ' ' if self.use_symbols else ''
        
        # Modify the format based on log level
        if record.levelno == logging.INFO:
            self._style._fmt = f'{color}{symbol}%(message)s{Style.RESET_ALL}'
        else:
            # For warnings, errors, etc. include more details
            module_path = record.name
            self._style._fmt = f'{color}{symbol}[{levelname}] %(asctime)s [{module_path}] %(message)s{Style.RESET_ALL}'
        
        # Format the record with the modified format
        result = super().format(record)
        
        # Restore the original format
        self._style._fmt = original_format
        
        return result

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add console handler with our custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())
root_logger.addHandler(console_handler)

# Create file handler for persistent logs
import os
from datetime import datetime
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"isco_pipeline_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(file_handler)

# Get module logger
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="ISCO Classification Pipeline")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and use existing model"
    )
    
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune existing model with corrections"
    )
    
    parser.add_argument(
        "--corrections-dir",
        type=str,
        default="data/corrections/",
        help="Directory with correction CSVs (default: data/corrections/)"
    )
    
    parser.add_argument(
        "--enable-optimizations",
        action="store_true",
        help="Enable mixed precision and gradient checkpointing"
    )
    
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate SHAP explanations for predictions"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input CSV file for prediction"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip the evaluation stage"
    )
    
    parser.add_argument(
        "--force-update-best",
        action="store_true",
        help="Force updating the best model even if the new model isn't better"
    )
    
    return parser.parse_args()

def load_and_update_config(args):
    """
    Load and update configuration based on arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        dict: Updated configuration
    """
    # Load config
    config = load_config(args.config)
    
    # Update optimization settings if flag is set
    if args.enable_optimizations:
        config["training"]["enable_optimizations"] = True
    
    return config

def run_preprocessing_and_training(config, force_update_best=False):
    """
    Run preprocessing and model training
    
    Args:
        config (dict): Configuration dictionary
        force_update_best (bool): Whether to force update the best model even if not better
        
    Returns:
        tuple: Trainer and label-to-ID mapping
    """
    # Preprocess data
    input_path = os.path.join(config["data"]["raw_dir"], "historical_records.csv")
    if os.path.exists(input_path):
        logger.info("Preprocessing data")
        preprocess_data(input_path, config)
    else:
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Add force_update_best flag to config
    config["training"]["force_update_best"] = force_update_best
    
    # Train model
    logger.info("Training model")
    trainer, label2id, code_metadata = train_model(config)
    
    return trainer, label2id, code_metadata

def run_fine_tuning(config, corrections_dir, force_update_best=False):
    """
    Fine-tune model with manual corrections
    
    Args:
        config (dict): Configuration dictionary
        corrections_dir (str): Directory with correction CSVs
        force_update_best (bool): Whether to force update the best model even if not better
    """
    # Check if corrections directory exists
    if not os.path.exists(corrections_dir):
        logger.error(f"Corrections directory not found: {corrections_dir}")
        raise FileNotFoundError(f"Corrections directory not found: {corrections_dir}")
    
    # Get all correction CSV files
    correction_files = [f for f in os.listdir(corrections_dir) if f.endswith(".csv")]
    
    if not correction_files:
        logger.warning(f"No correction CSV files found in {corrections_dir}")
        return
    
    # Load and combine all correction files
    correction_dfs = []
    for file in correction_files:
        file_path = os.path.join(corrections_dir, file)
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        if "text" not in df.columns or "corrected_isco_code" not in df.columns:
            logger.warning(f"Skipping {file}: missing required columns")
            continue
        
        # Rename column for consistency
        df = df.rename(columns={"corrected_isco_code": "isco_code"})
        correction_dfs.append(df[["text", "isco_code"]])
    
    if not correction_dfs:
        logger.warning("No valid correction files found")
        return
    
    # Combine all corrections
    corrections_df = pd.concat(correction_dfs, ignore_index=True)
    
    # Load original training data
    train_path = os.path.join(config["data"]["processed_dir"], "train.csv")
    train_df = pd.read_csv(train_path)
    
    # Combine with corrections
    combined_df = pd.concat([train_df, corrections_df], ignore_index=True)
    
    # Save combined data for fine-tuning
    fine_tune_dir = os.path.join(config["data"]["processed_dir"], "fine_tune")
    ensure_dir(fine_tune_dir)
    combined_df.to_csv(os.path.join(fine_tune_dir, "train.csv"), index=False)
    
    # Copy validation data
    val_path = os.path.join(config["data"]["processed_dir"], "val.csv")
    val_df = pd.read_csv(val_path)
    val_df.to_csv(os.path.join(fine_tune_dir, "val.csv"), index=False)
    
    # Update config for fine-tuning
    fine_tune_config = config.copy()
    fine_tune_config["data"]["processed_dir"] = fine_tune_dir
    fine_tune_config["model"]["epochs"] = 2
    fine_tune_config["model"]["learning_rate"] = 1.0e-5
    
    # Train model with combined data
    logger.info(f"Fine-tuning model with {len(corrections_df)} corrections")
    trainer, label2id, code_metadata = train_model(fine_tune_config)
    
    # Save fine-tuned model with code metadata only if it's better
    from src.model import save_best_model
    # Use force_update_best flag to determine whether to force save
    saved = save_best_model(trainer, config["output"]["best_model_dir"], 
                 label2id, code_metadata, force_save=force_update_best)
    
    if saved:
        logger.info(f"Saved fine-tuned model to {config['output']['best_model_dir']}")
    else:
        logger.warning("Fine-tuned model was not better than existing best model. Not updating best_model directory.")

# The load_isco_reference function has been moved to src/utils.py for reuse in both CLI and API

def run_prediction(input_path, config, explain=False):
    """
    Run prediction on input data
    
    Args:
        input_path (str): Path to input CSV
        config (dict): Configuration dictionary
        explain (bool): Whether to generate explanations
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load input data
    input_df = pd.read_csv(input_path)
    
    # Check if required columns exist
    required_columns = ["job_title", "duties_description"]
    missing_columns = [col for col in required_columns if col not in input_df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Preprocess input data
    from src.preprocess import clean_and_combine_text
    input_df["text"] = clean_and_combine_text(input_df["job_title"], input_df["duties_description"])
    
    # Rename input columns for standardized output
    input_df = input_df.rename(columns={
        "job_title": "title",
        "duties_description": "description"
    })
    
    # Load ISCO reference data for occupation titles
    reference_path = config["data"].get("reference_file")
    isco_code_to_title = load_isco_reference(reference_path)
    
    # Load model and mappings
    model, tokenizer, label_map = load_model_and_mappings(config["output"]["best_model_dir"])
    
    # Make predictions
    logger.info(f"Making predictions on {len(input_df)} records")
    predictions = predict_batch(
        input_df["text"].tolist(),
        model,
        tokenizer,
        label_map,
        config,
        explain=explain
    )
    
    # Add predictions to input DataFrame
    for i, pred in enumerate(predictions):
        predicted_code = pred["isco_code"]
        input_df.loc[i, "predicted_code"] = predicted_code
        input_df.loc[i, "confidence"] = pred["confidence"]
        input_df.loc[i, "confidence_grade"] = pred["confidence_grade"]
        input_df.loc[i, "is_fallback"] = pred["is_fallback"]
        
        # Add occupation title for predicted code if available
        if predicted_code in isco_code_to_title:
            input_df.loc[i, "predicted_occupation"] = isco_code_to_title[predicted_code]
        else:
            input_df.loc[i, "predicted_occupation"] = f"Occupation code {predicted_code}"
        
        # Add alternative predictions if available
        if "alternative_1" in pred:
            alt1_code = pred["alternative_1"]
            input_df.loc[i, "alternative_1"] = alt1_code
            input_df.loc[i, "alternative_1_confidence"] = pred["alternative_1_confidence"]
            
            # Add occupation title for alternative 1 if available
            if alt1_code in isco_code_to_title:
                input_df.loc[i, "alternative_1_occupation"] = isco_code_to_title[alt1_code]
        
        if "alternative_2" in pred:
            alt2_code = pred["alternative_2"]
            input_df.loc[i, "alternative_2"] = alt2_code
            input_df.loc[i, "alternative_2_confidence"] = pred["alternative_2_confidence"]
            
            # Add occupation title for alternative 2 if available
            if alt2_code in isco_code_to_title:
                input_df.loc[i, "alternative_2_occupation"] = isco_code_to_title[alt2_code]
    
    # Save predictions
    output_path = os.path.join(
        config["data"]["processed_dir"],
        f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    )
    input_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

def main():
    """
    Main function to run the ISCO classification pipeline
    """
    start_time = datetime.now()
    
    # Print welcome banner
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'ISCO Classification Pipeline':^80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}\n")
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load and update config
        config = load_and_update_config(args)
        
        # Log hardware info
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Log memory info if possible
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"System memory: {mem.total/(1024**3):.1f} GB total, {mem.available/(1024**3):.1f} GB available")
        except ImportError:
            pass
        
        # Display pipeline stages
        stages = []
        if not args.skip_training:
            stages.append("Preprocessing and Training")
        if args.fine_tune:
            stages.append("Fine-tuning")
        if args.input:
            stages.append("Prediction")
        if not args.skip_evaluation:
            stages.append("Evaluation")
        
        logger.info(f"Pipeline will run these stages: {', '.join(stages)}")
        
        # Warn if force update flag is used
        if args.force_update_best:
            logger.warning(f"{Fore.YELLOW}⚠️ Force update best model flag is set! This will replace the best model even if the new model is worse.{Style.RESET_ALL}")
        
        # Run preprocessing and training if not skipped
        if not args.skip_training:
            logger.info(f"{Fore.CYAN}{Style.BRIGHT}Stage 1/4: Preprocessing and Training{Style.RESET_ALL}")
            run_preprocessing_and_training(config, force_update_best=args.force_update_best)
            logger.info(f"{Fore.CYAN}Preprocessing and training completed successfully ✓{Style.RESET_ALL}")
        
        # Fine-tune if requested
        if args.fine_tune:
            stage_num = 2 if not args.skip_training else 1
            total_stages = 4 if not args.skip_training else 3
            logger.info(f"{Fore.CYAN}{Style.BRIGHT}Stage {stage_num}/{total_stages}: Fine-tuning{Style.RESET_ALL}")
            run_fine_tuning(config, args.corrections_dir, force_update_best=args.force_update_best)
            logger.info(f"{Fore.CYAN}Fine-tuning completed successfully ✓{Style.RESET_ALL}")
        
        # Run prediction if input provided
        if args.input:
            stage_num = 3 if not args.skip_training and args.fine_tune else 2 if not args.skip_training or args.fine_tune else 1
            total_stages = 4 if not args.skip_training and args.fine_tune else 3 if not args.skip_training or args.fine_tune else 2
            logger.info(f"{Fore.CYAN}{Style.BRIGHT}Stage {stage_num}/{total_stages}: Prediction{Style.RESET_ALL}")
            run_prediction(args.input, config, args.explain)
            logger.info(f"{Fore.CYAN}Prediction completed successfully ✓{Style.RESET_ALL}")
        
        # Evaluate model if not skipped
        if not args.skip_evaluation:
            # Calculate stage numbers based on which stages are running
            stage_count = len(stages)
            stage_num = stages.index("Evaluation") + 1 if "Evaluation" in stages else 0
            
            logger.info(f"{Fore.CYAN}{Style.BRIGHT}Stage {stage_num}/{stage_count}: Evaluation{Style.RESET_ALL}")
            metrics = evaluate_model(config)
            logger.info(f"{Fore.CYAN}Evaluation completed successfully ✓{Style.RESET_ALL}")
        else:
            # Set metrics to None so we don't try to print them later
            metrics = None
        
        # Calculate elapsed time
        elapsed_time = datetime.now() - start_time
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Print completion message with summary
        print(f"\n{Fore.GREEN}{Style.BRIGHT}{'='*80}")
        print(f"{Fore.GREEN}{Style.BRIGHT}{'Pipeline Completed Successfully':^80}")
        print(f"{Fore.GREEN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
        
        # Print key metrics if available
        if 'metrics' in locals() and metrics:
            print(f"\n{Fore.CYAN}Key Metrics:{Style.RESET_ALL}")
            print(f"  • Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  • 3-digit Accuracy: {metrics.get('accuracy_3digit', 'N/A'):.4f}")
            print(f"  • Macro F1 Score: {metrics.get('macro_f1', 'N/A'):.4f}")
            print(f"  • Top-3 Accuracy: {metrics.get('top3_accuracy', 'N/A'):.4f}")
        
        # Display time info
        print(f"\n{Fore.CYAN}Time Information:{Style.RESET_ALL}")
        print(f"  • Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  • Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  • Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Display log location
        print(f"\n{Fore.CYAN}Full logs saved to:{Style.RESET_ALL} {log_file}")
        
    except KeyboardInterrupt:
        logger.warning(f"{Fore.YELLOW}Pipeline interrupted by user.{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}Pipeline was interrupted by the user.{Style.RESET_ALL}")
        # Return a non-zero exit code
        return 1
    except Exception as e:
        # Log the full exception with traceback
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        
        # Print user-friendly error message
        print(f"\n{Fore.RED}{Style.BRIGHT}{'='*80}")
        print(f"{Fore.RED}{Style.BRIGHT}{'Pipeline Failed':^80}")
        print(f"{Fore.RED}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        print(f"\nFor detailed error information, check the log file: {log_file}")
        
        # Return a non-zero exit code
        return 1
    
    return 0

if __name__ == "__main__":
    main()