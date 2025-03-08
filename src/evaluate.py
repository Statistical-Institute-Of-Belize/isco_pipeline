import os
import json
import pandas as pd
import numpy as np
import logging
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer

from .utils import load_config, ensure_dir
from .model import ISCODataset

# Configure logging
logger = logging.getLogger(__name__)

def load_code_metadata(model_dir):
    """
    Load ISCO code metadata if available
    
    Args:
        model_dir (str): Directory with model files
        
    Returns:
        tuple: 
            - Code metadata dictionary
            - Code hierarchy dictionary
            - Dictionary mapping codes to titles
    """
    code_metadata = {}
    code_hierarchy = {}
    code_to_title = {}
    
    # Try to load code metadata
    metadata_path = os.path.join(model_dir, "code_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                code_metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(code_metadata)} ISCO codes")
            
            # Extract code to title mapping
            for code, metadata in code_metadata.items():
                if "title" in metadata:
                    code_to_title[code] = metadata["title"]
        except Exception as e:
            logger.warning(f"Error loading code metadata: {e}")
    
    # Try to load code hierarchy
    hierarchy_path = os.path.join(model_dir, "code_hierarchy.json")
    if os.path.exists(hierarchy_path):
        try:
            with open(hierarchy_path, "r") as f:
                code_hierarchy = json.load(f)
            logger.info(f"Loaded ISCO code hierarchy with {len(code_hierarchy.get('1', {}))} major groups")
        except Exception as e:
            logger.warning(f"Error loading code hierarchy: {e}")
    
    # If we don't have code metadata but have a reference file, try to load it
    if not code_metadata:
        try:
            from .model import load_isco_reference
            _, _, ref_code_to_title = load_isco_reference()
            if ref_code_to_title:
                code_to_title = ref_code_to_title
                logger.info(f"Loaded {len(code_to_title)} code titles from ISCO reference")
        except Exception as e:
            logger.debug(f"Could not load ISCO reference: {e}")
    
    return code_metadata, code_hierarchy, code_to_title

def load_test_data(config):
    """
    Load test data and label mapping
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: Test DataFrame, label-to-ID mapping, and code metadata
    """
    # Load test data
    test_path = os.path.join(config["data"]["processed_dir"], "test.csv")
    
    # Handle the case where fine-tuning data doesn't have test.csv
    try:
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
        else:
            # If in fine-tune mode and no test data exists, use original test data
            if "fine_tune" in config["data"]["processed_dir"]:
                original_dir = os.path.dirname(config["data"]["processed_dir"])
                original_test_path = os.path.join(original_dir, "test.csv")
                if os.path.exists(original_test_path):
                    test_df = pd.read_csv(original_test_path)
                    logger.info(f"Using test data from original directory: {original_test_path}")
                else:
                    # Create an empty test dataframe as fallback
                    logger.warning("No test data found. Creating empty test dataframe.")
                    test_df = pd.DataFrame(columns=["text", "isco_code"])
            else:
                # Create an empty test dataframe as fallback
                logger.warning("No test data found. Creating empty test dataframe.")
                test_df = pd.DataFrame(columns=["text", "isco_code"])
    except Exception as e:
        logger.warning(f"Error loading test data: {e}. Creating empty test dataframe.")
        test_df = pd.DataFrame(columns=["text", "isco_code"])
    
    # Load label mapping
    try:
        # Read label mapping from JSON
        label2id_path = os.path.join(config["output"]["best_model_dir"], "label2id.json")
        if os.path.exists(label2id_path):
            with open(label2id_path, "r") as f:
                label2id = json.load(f)
                logger.info(f"Loaded label mapping with {len(label2id)} entries")
                
                # Check if mapping looks valid (quick sanity check)
                if label2id:
                    sample_keys = list(label2id.keys())[:5]
                    sample_values = [label2id[k] for k in sample_keys]
                    logger.info(f"Sample label2id mapping: {dict(zip(sample_keys, sample_values))}")
                    
                    # Check if the values look like ISCO codes (numeric, 1-4 digits)
                    valid_values = all(str(v).isdigit() and 1 <= len(str(v)) <= 4 for v in label2id.values())
                    if not valid_values:
                        logger.warning("label2id mapping contains non-ISCO-like values")
                        
                        # Check if keys and values need to be swapped (sometimes it happens)
                        if all(k.isdigit() and 1 <= len(k) <= 4 for k in label2id.keys()):
                            logger.warning("Keys look like ISCO codes while values look like indices. The mapping may be inverted.")
        else:
            logger.warning(f"Label mapping file not found at {label2id_path}")
            label2id = {}
            
        # Verify all ISCO codes in test data are in mapping
        if label2id:
            # Check if we're working with classic label2id (ISCO code -> index) or inverted (index -> ISCO code) mapping
            # In the standard format, the keys are ISCO codes
            standard_format = all(str(k).isdigit() and 1 <= len(str(k)) <= 4 for k in label2id.keys())
            
            # Get the set of ISCO codes in our mapping (either keys or values depending on format)
            if standard_format:
                mapped_codes = set(str(k) for k in label2id.keys())
            else:
                mapped_codes = set(str(v) for v in label2id.values())
            
            # Check if we're using only 4-digit codes
            four_digit_only = True
            if standard_format:
                four_digit_only = all(len(str(k)) == 4 for k in label2id.keys())
            else:
                four_digit_only = all(len(str(v)) == 4 for v in label2id.values() if str(v).isdigit())
            
            # Log the mapping type we detected
            if four_digit_only:
                logger.info("Detected model trained with 4-digit ISCO codes only")
            else:
                logger.info("Detected model trained with all ISCO code levels (1-4 digits)")
            
            # Identify ISCO codes in test data that aren't in our mapping
            unknown_codes = []
            for code in test_df["isco_code"].unique():
                code_str = str(code)
                
                # For 4-digit-only models, skip non-4-digit codes in test data
                if four_digit_only and len(code_str) != 4:
                    continue
                    
                if code_str not in mapped_codes:
                    unknown_codes.append(code_str)
            
            if unknown_codes:
                if four_digit_only:
                    logger.warning(f"Found {len(unknown_codes)} 4-digit ISCO codes in test data that are not in the model's label mapping")
                else:
                    logger.warning(f"Found {len(unknown_codes)} ISCO codes in test data that are not in the model's label mapping")
                    
                if len(unknown_codes) <= 10:
                    logger.warning(f"Unknown codes: {', '.join(unknown_codes)}")
                else:
                    logger.warning(f"First 10 unknown codes: {', '.join(unknown_codes[:10])}")
            else:
                logger.info("All valid ISCO codes in test data are present in the model's label mapping")
    except Exception as e:
        logger.error(f"Error loading label mapping: {e}")
        logger.debug(f"Exception details: {traceback.format_exc()}")
        # Create empty mapping as fallback
        label2id = {}
    
    # Load code metadata
    code_metadata, code_hierarchy, code_to_title = load_code_metadata(config["output"]["best_model_dir"])
    
    return test_df, label2id, code_metadata, code_hierarchy, code_to_title

def prepare_dataset(test_df, label2id, max_seq_length):
    """
    Prepare test dataset
    
    Args:
        test_df (DataFrame): Test data
        label2id (dict): Label-to-ID mapping
        max_seq_length (int): Maximum sequence length
        
    Returns:
        ISCODataset: Test dataset
    """
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Convert labels to IDs with handling for unknown labels
    test_texts = test_df["text"].tolist()
    test_labels = []
    unknown_labels = set()
    filtered_texts = []
    
    # Find most common label to use as default
    most_common_label = 0
    try:
        # Try to determine most common label from label2id values
        if label2id:
            most_common_label = max(label2id.values())
    except:
        # If that fails, just use 0
        most_common_label = 0
    
    # Detect if we have a 4-digit-only model by looking at all keys that are ISCO codes
    isco_keys = [k for k in label2id.keys() if str(k).isdigit()]
    valid_length_keys = [k for k in isco_keys if len(str(k)) == 4 or (len(str(k)) < 4 and str(k).startswith('0'))]
    four_digit_only = len(valid_length_keys) == len(isco_keys)
    
    # Helper function to normalize ISCO codes to 4 digits
    def normalize_isco_code(code):
        code_str = str(code)
        # If it's already 4 digits, return as is
        if len(code_str) == 4:
            return code_str
        # If it's shorter and potentially a code with leading zeros
        elif len(code_str) < 4:
            # First check if zero-padded version exists in mapping
            padded = code_str.zfill(4)
            if padded in label2id:
                return padded
            # For specifically known armed forces codes
            if code_str in ['110', '210', '310']:
                armed_forces = '0' + code_str  # Convert to proper armed forces code
                if armed_forces in label2id:
                    return armed_forces
        # Return original if no normalization needed/possible
        return code_str
    
    # Process each label, handling normalization
    for i, label in enumerate(test_df["isco_code"].tolist()):
        # Try to normalize the code to handle potential leading zeros
        label_str = normalize_isco_code(label)
        
        # Skip non-4-digit codes if we're working with a 4-digit-only model
        if four_digit_only and len(label_str) != 4:
            continue
            
        # Check if this code is in our mapping
        if label_str in label2id:
            test_labels.append(label2id[label_str])
            filtered_texts.append(test_texts[i])
        else:
            # Last resort: try zero-padding if this is a shorter code
            if len(label_str) < 4:
                padded = label_str.zfill(4)
                if padded in label2id:
                    test_labels.append(label2id[padded])
                    filtered_texts.append(test_texts[i])
                    continue
            
            # If still not found, it's truly unknown
            unknown_labels.add(label_str)
    
    # Log warning if unknown labels were found
    if unknown_labels:
        if four_digit_only:
            logger.warning(f"Found {len(unknown_labels)} unknown 4-digit ISCO codes in test data during dataset preparation")
        else:
            logger.warning(f"Found {len(unknown_labels)} unknown ISCO codes in test data during dataset preparation")
            
        # Only show the first 10 unknown codes to keep logs clean
        if len(unknown_labels) <= 10:
            logger.warning(f"Unknown codes: {', '.join(sorted(unknown_labels))}")
        else:
            sorted_labels = sorted(unknown_labels)
            logger.warning(f"First 10 unknown codes: {', '.join(sorted_labels[:10])}")
            
        logger.warning(f"Filtered out {len(test_texts) - len(filtered_texts)} test samples with unknown labels")
    
    # If no valid labels remain, create a dummy dataset to avoid errors
    if not test_labels:
        logger.warning("No test samples with known labels. Creating dummy test dataset.")
        test_labels = [most_common_label]
        filtered_texts = ["dummy test sample"]
    
    # Create dataset with filtered data
    test_dataset = ISCODataset(filtered_texts, test_labels, tokenizer, max_seq_length)
    
    return test_dataset

def compute_metrics(predictions, label_ids, id2label):
    """
    Compute evaluation metrics
    
    Args:
        predictions (np.ndarray): Model predictions
        label_ids (np.ndarray): True labels
        id2label (dict): ID-to-label mapping
        
    Returns:
        dict: Dictionary with metrics
    """
    # Get predicted classes
    preds = np.argmax(predictions, axis=1)
    
    # First, check the format of id2label mapping
    # It should have string keys, but sometimes it might have integer keys
    # Ensure consistent format (string keys)
    id2label_str = {}
    for k, v in id2label.items():
        id2label_str[str(k)] = str(v)
    id2label = id2label_str
    
    # Check for missing IDs
    unique_label_ids = set(map(int, label_ids))
    unique_pred_ids = set(map(int, preds))
    all_ids = unique_label_ids.union(unique_pred_ids)
    
    missing_ids = [i for i in all_ids if str(i) not in id2label]
    if missing_ids:
        logger.warning(f"Found {len(missing_ids)} IDs missing from id2label mapping")
        
        # Log some examples of missing IDs for debugging
        if missing_ids:
            sample_missing = missing_ids[:5] if len(missing_ids) > 5 else missing_ids
            logger.info(f"Examples of missing IDs: {sample_missing}")
            logger.info(f"Available ID range: {min(map(int, id2label.keys()))} to {max(map(int, id2label.keys()))}")
        
        # Create temporary mapping for these IDs to avoid errors
        for i in missing_ids:
            id2label[str(i)] = f"unknown_{i}"
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(label_ids, preds),
        "macro_f1": f1_score(label_ids, preds, average="macro"),
        "weighted_f1": f1_score(label_ids, preds, average="weighted"),
    }
    
    # Compute top-3 accuracy
    top3_indices = np.argsort(-predictions, axis=1)[:, :3]
    top3_accuracy = np.mean([1 if label_ids[i] in top3_indices[i] else 0 for i in range(len(label_ids))])
    metrics["top3_accuracy"] = top3_accuracy
    
    try:
        # Convert label IDs to ISCO codes
        true_codes = []
        unknown_ids = set()
        
        for label_id in label_ids:
            label_key = str(int(label_id))  # Convert to int then string to normalize
            if label_key in id2label:
                true_codes.append(id2label[label_key])
            else:
                unknown_ids.add(label_key)
                true_codes.append("0000")  # Default placeholder
        
        if unknown_ids:
            logger.warning(f"Found {len(unknown_ids)} unique unknown label IDs out of {len(label_ids)} total labels")
        
        pred_codes = []
        unknown_preds = set()
        
        for pred in preds:
            pred_key = str(int(pred))  # Convert to int then string to normalize
            if pred_key in id2label:
                pred_codes.append(id2label[pred_key])
            else:
                unknown_preds.add(pred_key)
                pred_codes.append("0000")  # Default placeholder
        
        if unknown_preds:
            logger.warning(f"Found {len(unknown_preds)} unique unknown prediction IDs out of {len(preds)} total predictions")
        
        # Get 3-digit versions (if a code is shorter than 3 digits, we'll use the full code)
        true_3digit = [code[:3] if len(code) >= 3 else code for code in true_codes]
        pred_3digit = [code[:3] if len(code) >= 3 else code for code in pred_codes]
        
        # Compute 3-digit accuracy and F1
        metrics["accuracy_3digit"] = accuracy_score(true_3digit, pred_3digit)
        metrics["macro_f1_3digit"] = f1_score(true_3digit, pred_3digit, average="macro")
        
        # Add the number of unique codes evaluated
        metrics["unique_true_codes"] = len(set(true_codes))
        metrics["unique_pred_codes"] = len(set(pred_codes))
        
    except Exception as e:
        logger.warning(f"Failed to compute 3-digit metrics: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        metrics["accuracy_3digit"] = 0.0
        metrics["macro_f1_3digit"] = 0.0
    
    return metrics

def generate_html_report(metrics, error_dir, date_str, code_to_title=None, top_errors=None, major_group_errors=None):
    """
    Generate an HTML report summarizing evaluation metrics
    
    Args:
        metrics (dict): Evaluation metrics
        error_dir (str): Error analysis directory
        date_str (str): Date string for the report
        code_to_title (dict, optional): Mapping from ISCO codes to titles
        top_errors (DataFrame, optional): Top misclassifications
        major_group_errors (DataFrame, optional): Error rates by major group
    """
    # Check which visualization is available
    confusion_matrix_path = os.path.join(error_dir, "confusion_matrix_major_groups.png")
    single_group_path = os.path.join(error_dir, "major_group_accuracy.png")
    details_path = os.path.join(error_dir, "single_group_details.txt")
    
    # Create the confusion matrix or single group visualization HTML
    confusion_matrix_html = ""
    if os.path.exists(confusion_matrix_path):
        confusion_matrix_html = f"""
        <p>The confusion matrix shows how predictions are distributed across major groups (1-digit ISCO codes).</p>
        <p>Each row represents the true major group, and each column represents the predicted major group. The values are normalized by row, showing what percentage of each true class was classified as each predicted class.</p>
        <div class="chart-container">
            <img src="confusion_matrix_major_groups.png" alt="Confusion Matrix">
        </div>
        """
    elif os.path.exists(single_group_path):
        details_html = ""
        if os.path.exists(details_path):
            with open(details_path, 'r') as f:
                details_content = f.read()
                details_html = f"""
                <h3>Details:</h3>
                <pre class="code-block">
{details_content}
                </pre>
                """
        
        confusion_matrix_html = f"""
        <p>All predictions fall within a single major group. Instead of a confusion matrix, here's the accuracy for this group:</p>
        <div class="chart-container">
            <img src="major_group_accuracy.png" alt="Major Group Accuracy">
        </div>
        {details_html}
        """
    else:
        confusion_matrix_html = "<p>No major group analysis visualization available.</p>"
    # Create metrics HTML
    metrics_html = f"""
    <div class="metrics-container">
        <div class="metric-card">
            <h3>Accuracy</h3>
            <div class="metric-value">{metrics["accuracy"]:.4f}</div>
            <p>Percentage of correctly classified occupations</p>
        </div>
        <div class="metric-card">
            <h3>Top-3 Accuracy</h3>
            <div class="metric-value">{metrics["top3_accuracy"]:.4f}</div>
            <p>Percentage where correct label is in top 3 predictions</p>
        </div>
        <div class="metric-card">
            <h3>Macro F1 Score</h3>
            <div class="metric-value">{metrics["macro_f1"]:.4f}</div>
            <p>Average F1 score across all classes (unweighted)</p>
        </div>
        <div class="metric-card">
            <h3>Weighted F1 Score</h3>
            <div class="metric-value">{metrics["weighted_f1"]:.4f}</div>
            <p>Average F1 score weighted by class frequency</p>
        </div>
        <div class="metric-card">
            <h3>3-Digit Accuracy</h3>
            <div class="metric-value">{metrics["accuracy_3digit"]:.4f}</div>
            <p>Accuracy when considering only first 3 digits of ISCO code</p>
        </div>
        <div class="metric-card">
            <h3>3-Digit Macro F1</h3>
            <div class="metric-value">{metrics["macro_f1_3digit"]:.4f}</div>
            <p>Macro F1 score for 3-digit classification</p>
        </div>
    </div>
    """
    
    # Generate top errors HTML if available
    top_errors_html = ""
    if top_errors is not None and len(top_errors) > 0:
        top_errors_rows = []
        for _, row in top_errors.iterrows():
            true_code = str(row['true_code'])
            pred_code = str(row['predicted_code'])
            count = row['count']
            
            # Get titles if available
            true_title = code_to_title.get(true_code, f"Code {true_code}") if code_to_title else f"Code {true_code}"
            pred_title = code_to_title.get(pred_code, f"Code {pred_code}") if code_to_title else f"Code {pred_code}"
            
            top_errors_rows.append(f"""
            <tr>
                <td>{true_code}</td>
                <td>{true_title}</td>
                <td>{pred_code}</td>
                <td>{pred_title}</td>
                <td>{count}</td>
            </tr>
            """)
        
        top_errors_table = "\n".join(top_errors_rows)
        top_errors_html = f"""
        <h3>Top Misclassifications</h3>
        <table>
            <thead>
                <tr>
                    <th>True Code</th>
                    <th>True Occupation</th>
                    <th>Predicted Code</th>
                    <th>Predicted Occupation</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                {top_errors_table}
            </tbody>
        </table>
        """
    
    # Generate major group errors HTML if available
    major_group_html = ""
    if major_group_errors is not None and len(major_group_errors) > 0:
        major_group_rows = []
        
        # Define major group titles
        major_group_titles = {
            "0": "Armed Forces Occupations",
            "1": "Managers",
            "2": "Professionals",
            "3": "Technicians and Associate Professionals",
            "4": "Clerical Support Workers",
            "5": "Service and Sales Workers",
            "6": "Skilled Agricultural, Forestry and Fishery Workers",
            "7": "Craft and Related Trades Workers",
            "8": "Plant and Machine Operators and Assemblers",
            "9": "Elementary Occupations"
        }
        
        for _, row in major_group_errors.iterrows():
            group = str(row['true_major_group'])
            error_rate = row['error_rate']
            
            # Get title if available
            title = major_group_titles.get(group, f"Major Group {group}")
            
            # Calculate accuracy
            accuracy = 1 - error_rate
            
            # Determine color based on accuracy
            color = "#cc0000" if accuracy < 0.6 else "#ff9900" if accuracy < 0.8 else "#009900"
            
            major_group_rows.append(f"""
            <tr>
                <td>{group}</td>
                <td>{title}</td>
                <td><div class="progress-bar" style="width: {accuracy*100}%; background-color: {color};">{accuracy:.2f}</div></td>
                <td>{error_rate:.4f}</td>
            </tr>
            """)
        
        major_group_table = "\n".join(major_group_rows)
        major_group_html = f"""
        <h3>Performance by Major Group</h3>
        <table>
            <thead>
                <tr>
                    <th>Major Group</th>
                    <th>Title</th>
                    <th>Accuracy</th>
                    <th>Error Rate</th>
                </tr>
            </thead>
            <tbody>
                {major_group_table}
            </tbody>
        </table>
        """
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ISCO Classification Model Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .metrics-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .metric-card {{
                background-color: #f5f5f5;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                width: 45%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #0066cc;
            }}
            .section {{
                margin: 30px 0;
            }}
            h1 {{
                color: #333;
            }}
            h2 {{
                color: #444;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }}
            h3 {{
                color: #555;
                margin-top: 20px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 20px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 3px rgba(0,0,0,0.1);
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
                border-bottom: 2px solid #ddd;
            }}
            tr:hover {{
                background-color: #f9f9f9;
            }}
            .progress-bar {{
                height: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                font-weight: bold;
                line-height: 20px;
            }}
            .navigation {{
                position: sticky;
                top: 0;
                background-color: white;
                padding: 10px 0;
                border-bottom: 1px solid #ddd;
                margin-bottom: 20px;
                z-index: 100;
            }}
            .navigation a {{
                display: inline-block;
                margin-right: 15px;
                text-decoration: none;
                color: #0066cc;
                padding: 5px 10px;
                border-radius: 4px;
            }}
            .navigation a:hover {{
                background-color: #f0f0f0;
            }}
            .chart-container {{
                margin: 20px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 15px;
                background-color: #fff;
            }}
        </style>
    </head>
    <body>
        <div class="navigation">
            <a href="#overview">Overview</a>
            <a href="#metrics">Metrics</a>
            <a href="#errors">Error Analysis</a>
            <a href="#confusion">Confusion Matrix</a>
            <a href="#major-groups">Major Groups</a>
            <a href="#conclusion">Conclusion</a>
        </div>
    
        <div class="header">
            <h1>ISCO Classification Model Evaluation Report</h1>
            <p>Generated on: {date_str}</p>
        </div>

        <div id="overview" class="section">
            <h2>Overview</h2>
            <p>This report evaluates the performance of an ISCO code classification model trained on occupation descriptions.</p>
            <p>The International Standard Classification of Occupations (ISCO) is a tool for organizing jobs into a clearly defined set of groups according to the tasks and duties undertaken in the job. It is maintained by the International Labour Organization (ILO).</p>
            <p>The model is evaluated using {metrics.get("evaluation_samples", "N/A")} test samples across various ISCO codes at different levels (1-4 digits).</p>
        </div>

        <div id="metrics" class="section">
            <h2>Performance Metrics</h2>
            {metrics_html}
        </div>

        <div id="errors" class="section">
            <h2>Error Analysis</h2>
            <p>For detailed error analysis, check the following files in the error_analysis directory:</p>
            <ul>
                <li><a href="full_results.csv">Full Results (CSV)</a></li>
                <li><a href="top_misclassifications.csv">Top Misclassifications (CSV)</a></li>
                <li><a href="error_rates_by_major_group.csv">Error Rates by Major Group (CSV)</a></li>
                <li><a href="error_rates_by_submajor_group.csv">Error Rates by Sub-Major Group (CSV)</a></li>
            </ul>
            
            {top_errors_html}
        </div>
        
        <div id="confusion" class="section">
            <h2>Major Group Analysis</h2>
            {confusion_matrix_html}
        </div>
        
        <div id="major-groups" class="section">
            <h2>Performance by Major Group</h2>
            {major_group_html}
        </div>
        
        <div id="conclusion" class="section">
            <h2>Conclusion</h2>
            <p>This report summarizes the performance of the ISCO classification model. The model achieves:</p>
            <ul>
                <li>An overall accuracy of {metrics["accuracy"]:.4f}</li>
                <li>A 3-digit accuracy of {metrics["accuracy_3digit"]:.4f}</li>
                <li>A top-3 accuracy of {metrics["top3_accuracy"]:.4f}</li>
            </ul>
            <p>For further improvements, consider:</p>
            <ul>
                <li>Adding more training data for underperforming classes</li>
                <li>Fine-tuning model hyperparameters</li>
                <li>Implementing data augmentation techniques</li>
                <li>Using hierarchical classification approaches</li>
                <li>Exploring transfer learning with domain-specific pretraining</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    report_path = os.path.join(error_dir, "evaluation_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    return report_path

def perform_error_analysis(predictions, label_ids, test_df, id2label, output_dir, code_to_title=None):
    """
    Perform error analysis and save results
    
    Args:
        predictions (np.ndarray): Model predictions
        label_ids (np.ndarray): True labels
        test_df (DataFrame): Test data
        id2label (dict): ID-to-label mapping
        output_dir (str): Directory to save error analysis
        code_to_title (dict, optional): Mapping from ISCO codes to titles
        
    Returns:
        tuple: Error analysis directory path and date string
    """
    # Get predicted classes
    preds = np.argmax(predictions, axis=1)
    
    # Create error analysis directory
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_dir = os.path.join(output_dir, f"run_{date_str}", "error_analysis")
    ensure_dir(error_dir)
    
    # Filter test_df to match the filtered dataset size
    # This is important if we filtered out unknown labels during prepare_dataset
    if len(preds) != len(test_df):
        logger.warning(f"Size mismatch between predictions ({len(preds)}) and test data ({len(test_df)}). Using only valid examples.")
        # Create a new DataFrame with only valid predictions
        valid_indices = []
        unknown_labels = set()
        
        # Determine which rows had valid labels
        for i, label in enumerate(test_df["isco_code"].tolist()):
            label_str = str(label)
            if label_str in id2label.values():
                valid_indices.append(i)
            else:
                unknown_labels.add(label_str)
        
        # Only keep rows with valid indices if we have enough valid indices
        if len(valid_indices) >= len(preds):
            test_df = test_df.iloc[valid_indices[:len(preds)]].reset_index(drop=True)
        else:
            # If we don't have enough valid indices, create a dummy DataFrame
            logger.warning("Not enough valid examples for error analysis. Using dummy data.")
            test_df = pd.DataFrame({"text": ["dummy example"] * len(preds), "isco_code": ["0000"] * len(preds)})
    
    # Create results DataFrame
    results_df = test_df.copy()
    
    # Create lookup dictionaries to track missing IDs
    missing_pred_ids = set()
    missing_true_ids = set()
    
    # Safely map predicted and true labels
    predicted_codes = []
    for pred in preds:
        pred_key = str(pred)
        if pred_key in id2label:
            # Get the ISCO code from the mapping
            isco_code = id2label[pred_key]
            # Check if it looks like a valid ISCO code
            if not isco_code or not isco_code.isdigit():
                logger.warning(f"Invalid ISCO code '{isco_code}' for ID {pred_key}")
                isco_code = "0000"  # Default placeholder
            predicted_codes.append(isco_code)
        else:
            missing_pred_ids.add(pred_key)
            predicted_codes.append("0000")  # Default placeholder
    
    true_codes = []
    for label_id in label_ids:
        label_key = str(label_id)
        if label_key in id2label:
            # Get the ISCO code from the mapping
            isco_code = id2label[label_key]
            # Check if it looks like a valid ISCO code
            if not isco_code or not isco_code.isdigit():
                logger.warning(f"Invalid ISCO code '{isco_code}' for ID {label_key}")
                isco_code = "0000"  # Default placeholder
            true_codes.append(isco_code)
        else:
            missing_true_ids.add(label_key)
            true_codes.append("0000")  # Default placeholder
    
    # Log summary of missing IDs after processing all items
    if missing_pred_ids:
        logger.warning(f"Found {len(missing_pred_ids)} unique prediction IDs not in mapping. Examples: {list(missing_pred_ids)[:5]}")
    if missing_true_ids:
        logger.warning(f"Found {len(missing_true_ids)} unique true label IDs not in mapping. Examples: {list(missing_true_ids)[:5]}")
    
    # Ensure we have the right number of codes
    if len(predicted_codes) < len(results_df):
        predicted_codes.extend(["0000"] * (len(results_df) - len(predicted_codes)))
    if len(true_codes) < len(results_df):
        true_codes.extend(["0000"] * (len(results_df) - len(true_codes)))
        
    results_df["predicted_code"] = predicted_codes[:len(results_df)]
    results_df["true_code"] = true_codes[:len(results_df)]
    results_df["correct"] = results_df["predicted_code"] == results_df["true_code"]
    
    # Add code titles if available
    if code_to_title:
        results_df["true_title"] = results_df["true_code"].apply(
            lambda code: code_to_title.get(str(code), f"Code {code}")
        )
        results_df["predicted_title"] = results_df["predicted_code"].apply(
            lambda code: code_to_title.get(str(code), f"Code {code}")
        )
    
    # Save full results
    results_df.to_csv(os.path.join(error_dir, "full_results.csv"), index=False)
    
    # Find misclassifications
    misclassifications = results_df[~results_df["correct"]].copy()
    
    # Top 10 misclassifications
    top_misclass = misclassifications.groupby(["true_code", "predicted_code"]).size().reset_index(name="count")
    top_misclass = top_misclass.sort_values("count", ascending=False).head(10)
    
    # Add titles to top misclassifications if available
    if code_to_title:
        top_misclass["true_title"] = top_misclass["true_code"].apply(
            lambda code: code_to_title.get(str(code), f"Code {code}")
        )
        top_misclass["predicted_title"] = top_misclass["predicted_code"].apply(
            lambda code: code_to_title.get(str(code), f"Code {code}")
        )
    
    top_misclass.to_csv(os.path.join(error_dir, "top_misclassifications.csv"), index=False)
    
    # Error rates by major group (1-digit)
    # Extract the major group (first digit of ISCO code)
    results_df["true_major_group"] = results_df["true_code"].str[0]
    results_df["predicted_major_group"] = results_df["predicted_code"].str[0]
    
    # Log unique major groups for debugging
    unique_true_major = results_df["true_major_group"].unique()
    unique_pred_major = results_df["predicted_major_group"].unique()
    logger.info(f"Unique true major groups: {sorted(unique_true_major)}")
    logger.info(f"Unique predicted major groups: {sorted(unique_pred_major)}")
    
    # Check for possible mapping issues
    if len(unique_true_major) <= 1 or len(unique_pred_major) <= 1:
        # There might be an issue with the ISCO codes - let's verify some samples
        sample_true = results_df["true_code"].head(5).tolist()
        sample_pred = results_df["predicted_code"].head(5).tolist()
        logger.warning(f"Possible ISCO code issue - Sample true codes: {sample_true}")
        logger.warning(f"Possible ISCO code issue - Sample predicted codes: {sample_pred}")
        
        # Try to debug the mapping
        logger.info(f"Check if predicted codes are valid: {all(len(code) > 0 for code in results_df['predicted_code'])}")
        logger.info(f"Check if true codes are valid: {all(len(code) > 0 for code in results_df['true_code'])}")
    
    # Calculate error rates by major group
    major_group_errors = results_df.groupby("true_major_group").apply(
        lambda x: 1 - x["correct"].mean()
    ).reset_index(name="error_rate")
    
    # Add sample counts for major groups
    major_group_counts = results_df.groupby("true_major_group").size().reset_index(name="sample_count")
    major_group_errors = major_group_errors.merge(major_group_counts, on="true_major_group")
    
    # Add titles for major groups
    major_group_titles = {
        "0": "Armed Forces Occupations",
        "1": "Managers",
        "2": "Professionals",
        "3": "Technicians and Associate Professionals",
        "4": "Clerical Support Workers",
        "5": "Service and Sales Workers",
        "6": "Skilled Agricultural, Forestry and Fishery Workers",
        "7": "Craft and Related Trades Workers",
        "8": "Plant and Machine Operators and Assemblers",
        "9": "Elementary Occupations"
    }
    
    major_group_errors["group_title"] = major_group_errors["true_major_group"].map(major_group_titles)
    major_group_errors.to_csv(os.path.join(error_dir, "error_rates_by_major_group.csv"), index=False)
    
    # Error rates by sub-major group (2-digit)
    results_df["true_submajor_group"] = results_df["true_code"].str[:2]
    results_df["predicted_submajor_group"] = results_df["predicted_code"].str[:2]
    
    submajor_group_errors = results_df.groupby("true_submajor_group").apply(
        lambda x: 1 - x["correct"].mean()
    ).reset_index(name="error_rate")
    
    # Add sample counts for sub-major groups
    submajor_group_counts = results_df.groupby("true_submajor_group").size().reset_index(name="sample_count")
    submajor_group_errors = submajor_group_errors.merge(submajor_group_counts, on="true_submajor_group")
    
    # Add titles for sub-major groups if available
    if code_to_title:
        submajor_group_errors["group_title"] = submajor_group_errors["true_submajor_group"].apply(
            lambda code: code_to_title.get(str(code), f"Sub-Major Group {code}")
        )
    
    submajor_group_errors.to_csv(os.path.join(error_dir, "error_rates_by_submajor_group.csv"), index=False)
    
    # Create confusion matrix heatmap for major groups
    plt.figure(figsize=(12, 10))
    
    # Check the distribution of major groups
    true_major_dist = results_df["true_major_group"].value_counts()
    pred_major_dist = results_df["predicted_major_group"].value_counts()
    
    logger.info(f"True major group distribution: {dict(true_major_dist)}")
    logger.info(f"Predicted major group distribution: {dict(pred_major_dist)}")
    
    # Check if codes look valid (not too many 0's)
    if "0" in true_major_dist and true_major_dist["0"] > len(results_df) * 0.5:
        logger.warning(f"More than 50% of true codes start with '0', which is unusual for ISCO. Check mappings.")
    if "0" in pred_major_dist and pred_major_dist["0"] > len(results_df) * 0.5:
        logger.warning(f"More than 50% of predicted codes start with '0', which is unusual for ISCO. Check mappings.")
    
    # Get all possible major groups (both true and predicted)
    all_major_groups = sorted(set(
        results_df["true_major_group"].unique().tolist() + 
        results_df["predicted_major_group"].unique().tolist()
    ))
    
    # Handle the single label case
    if len(all_major_groups) <= 1:
        logger.warning("Only one major group found, cannot create meaningful confusion matrix")
        
        # Instead of just a 1x1 matrix, let's create a more informative visualization
        plt.figure(figsize=(10, 6))
        
        # Get the single group or use a default
        major_group = all_major_groups[0] if all_major_groups else "0"
        group_title = major_group_titles.get(major_group, f"Major Group {major_group}")
        
        # Create a summary bar chart showing accuracy
        accuracy = results_df["correct"].mean()
        colors = ["#4CAF50" if accuracy >= 0.7 else "#FFC107" if accuracy >= 0.5 else "#F44336"]
        
        plt.bar(["Accuracy"], [accuracy], color=colors, width=0.5)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"Accuracy for Major Group {major_group}: {group_title}")
        
        # Add value labels
        plt.text(0, accuracy + 0.02, f"{accuracy:.2f}", ha='center', fontweight='bold')
        
        # Add explanatory note
        plt.figtext(0.5, 0.01, 
                    "Note: All predictions fall within the same major group.\nNo confusion matrix available.",
                    ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(error_dir, "major_group_accuracy.png"), dpi=300)
        plt.close()
        
        # Also save a simple text file with the information
        with open(os.path.join(error_dir, "single_group_details.txt"), "w") as f:
            f.write(f"All predictions fall within Major Group {major_group}: {group_title}\n")
            f.write(f"Number of samples: {len(results_df)}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            
            # Add submajor group breakdown if available
            if "true_submajor_group" in results_df.columns:
                submajor_counts = results_df["true_submajor_group"].value_counts()
                f.write(f"\nSubmajor Group Distribution:\n")
                for group, count in submajor_counts.items():
                    group_title = code_to_title.get(group, f"Group {group}") if code_to_title else f"Group {group}"
                    f.write(f"  {group}: {group_title} - {count} samples\n")
    else:
        # Create confusion matrix with explicit labels to avoid warning
        cm = confusion_matrix(
            results_df["true_major_group"], 
            results_df["predicted_major_group"],
            normalize="true",
            labels=all_major_groups
        )
        major_groups = all_major_groups
        
        # Create heatmap with better labels
        major_group_labels = [f"{g} - {major_group_titles.get(g, '')[:20]}" for g in major_groups]
        
        # Create the visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=".2f", 
            cmap="Blues",
            xticklabels=major_group_labels,
            yticklabels=major_group_labels
        )
        plt.title("Normalized Confusion Matrix (Major Groups)")
        plt.xlabel("Predicted Major Group")
        plt.ylabel("True Major Group")
        plt.tight_layout()
        plt.savefig(os.path.join(error_dir, "confusion_matrix_major_groups.png"), dpi=300)
        plt.close()
    
    # Create examples of common errors
    for i, row in top_misclass.head(5).iterrows():
        true_code = row["true_code"]
        pred_code = row["predicted_code"]
        
        # Get examples of this error
        examples = misclassifications[
            (misclassifications["true_code"] == true_code) & 
            (misclassifications["predicted_code"] == pred_code)
        ].head(5)
        
        # Add prediction confidence if available (for classification reports)
        if "text" in examples.columns:
            # Prepare a table of examples with context
            examples_table = examples[["text", "true_code", "predicted_code"]].copy()
            
            # Add titles if available
            if code_to_title:
                examples_table["true_title"] = examples_table["true_code"].apply(
                    lambda code: code_to_title.get(str(code), f"Code {code}")
                )
                examples_table["predicted_title"] = examples_table["predicted_code"].apply(
                    lambda code: code_to_title.get(str(code), f"Code {code}")
                )
        
        # Save examples
        examples.to_csv(
            os.path.join(error_dir, f"error_examples_{true_code}_to_{pred_code}.csv"),
            index=False
        )
    
    # Create a summary of the error analysis
    # Convert numpy values to Python native types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return convert_for_json(obj.to_dict())
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        else:
            return obj
    
    # Prepare summary dict with proper type conversion
    summary = {
        "total_samples": int(len(results_df)),
        "correct_predictions": int(results_df["correct"].sum()),
        "incorrect_predictions": int(len(results_df) - results_df["correct"].sum()),
        "accuracy": float(results_df["correct"].mean()),
        "total_unique_codes": int(len(results_df["true_code"].unique())),
        "total_major_groups": int(len(results_df["true_major_group"].unique())),
    }
    
    # Add worst and best performing major groups if available
    if len(major_group_errors) > 0:
        worst_idx = major_group_errors["error_rate"].idxmax()
        best_idx = major_group_errors["error_rate"].idxmin()
        
        summary["worst_performing_major_group"] = convert_for_json(
            major_group_errors.loc[worst_idx].to_dict()
        )
        summary["best_performing_major_group"] = convert_for_json(
            major_group_errors.loc[best_idx].to_dict()
        )
    
    # Add top misclassifications if available
    if len(top_misclass) > 0:
        summary["top_5_misclassifications"] = convert_for_json(
            top_misclass.head(5).to_dict("records")
        )
    
    # Save summary
    with open(os.path.join(error_dir, "error_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return error_dir, date_str

def evaluate_model(config):
    """
    Evaluate model performance with metrics and error analysis
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary with metrics
    """
    # Load test data, label mapping, and code metadata
    test_df, label2id, code_metadata, code_hierarchy, code_to_title = load_test_data(config)
    
    # Create id2label mapping
    id2label = {v: k for k, v in label2id.items()}
    
    # Create output directory for metrics
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output"]["model_dir"], f"runs/run_{date_str}")
    ensure_dir(output_dir)
    
    # Skip evaluation if test dataset is empty
    if len(test_df) == 0:
        logger.warning("No test data available. Skipping evaluation.")
        
        # Create empty metrics
        metrics = {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "top3_accuracy": 0.0,
            "accuracy_3digit": 0.0,
            "macro_f1_3digit": 0.0,
            "evaluation_samples": 0
        }
        
        # Save empty metrics
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
    
    # Prepare dataset
    test_dataset = prepare_dataset(test_df, label2id, config["model"]["max_seq_length"])
    
    # Load model
    model_path = config["output"]["best_model_dir"]
    
    # Important: The model's config.json contains the correct id2label mapping it was trained with
    logger.info(f"Loading model from {model_path}")
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    
    # Ensure we're using the model's internal mapping for consistency (single source of truth)
    try:
        # First, try to get mapping directly from model config object
        if hasattr(model.config, 'id2label') and model.config.id2label:
            logger.info("Using label mapping from model's config object")
            # The model's mapping uses string keys representing the class ID
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            logger.info(f"Model has {len(id2label)} label mappings in config")
        else:
            # Fallback: try to read from config.json file
            logger.warning("Model config object missing id2label attribute, falling back to config.json file")
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                    
                if "id2label" in model_config:
                    logger.info("Using label mapping from config.json file")
                    id2label = {int(k): str(v) for k, v in model_config["id2label"].items()}
                    logger.info(f"Model has {len(id2label)} label mappings")
                else:
                    logger.warning("No id2label found in config.json")
                    id2label = {}
            else:
                # If config.json doesn't exist, fall back to the loaded mapping
                logger.warning(f"No config.json found in {model_path}, falling back to id2label.json")
                id2label = {}
                    
        # Second fallback: If model config mapping is empty, try id2label.json file
        if not id2label:
            logger.warning("No mapping from model config, falling back to id2label.json")
            id2label_path = os.path.join(model_path, "id2label.json")
            if os.path.exists(id2label_path):
                try:
                    with open(id2label_path, 'r') as f:
                        id2label = json.load(f)
                        # Convert keys to int for consistency
                        id2label = {int(k): str(v) for k, v in id2label.items()}
                        logger.info(f"Loaded {len(id2label)} mappings from id2label.json")
                except Exception as e:
                    logger.warning(f"Error loading id2label.json: {e}")
                    id2label = {}
            else:
                logger.error(f"No id2label.json found in {model_path}")
                id2label = {}
            
        # Convert id2label keys to strings for compatibility with existing code
        id2label = {str(k): v for k, v in id2label.items()}
        
        if not id2label:
            logger.error("Could not load label mappings from any source!")
            
    except Exception as e:
        logger.warning(f"Could not get label mapping from model config: {e}")
        logger.debug(f"Exception details: {traceback.format_exc()}")
        id2label = {}
    
    # Get device and move model to it
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create trainer
    trainer = Trainer(model=model)
    
    # Evaluate model
    logger.info("Evaluating model")
    predictions = trainer.predict(test_dataset)
    
    # Log some statistics about predictions and labels
    logger.info(f"Prediction tensor shape: {predictions.predictions.shape}")
    logger.info(f"Label IDs shape: {predictions.label_ids.shape}")
    
    # Get unique labels to check for potential issues
    unique_preds = np.unique(np.argmax(predictions.predictions, axis=1))
    unique_labels = np.unique(predictions.label_ids)
    logger.info(f"Found {len(unique_preds)} unique predicted classes and {len(unique_labels)} unique true labels")
    
    # Compute metrics
    metrics = compute_metrics(predictions.predictions, predictions.label_ids, id2label)
    
    # Add number of evaluation samples
    metrics["evaluation_samples"] = len(predictions.label_ids)
    
    # Print metrics
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{name}: {value:.4f}")
        else:
            logger.info(f"{name}: {value}")
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Perform error analysis
    try:
        error_dir, date_str = perform_error_analysis(
            predictions.predictions, 
            predictions.label_ids,
            test_df,
            id2label,
            config["output"]["model_dir"],
            code_to_title=code_to_title
        )
        
        # Load error analysis data for the HTML report
        top_errors_path = os.path.join(error_dir, "top_misclassifications.csv")
        major_group_errors_path = os.path.join(error_dir, "error_rates_by_major_group.csv")
        
        top_errors = None
        major_group_errors = None
        
        if os.path.exists(top_errors_path):
            top_errors = pd.read_csv(top_errors_path)
            
        if os.path.exists(major_group_errors_path):
            major_group_errors = pd.read_csv(major_group_errors_path)
        
        # Generate HTML report with additional metadata
        logger.info("Generating HTML evaluation report")
        report_path = generate_html_report(
            metrics, 
            error_dir, 
            date_str,
            code_to_title=code_to_title,
            top_errors=top_errors,
            major_group_errors=major_group_errors
        )
        logger.info(f"HTML report saved to: {report_path}")
        
        # Save code metadata alongside the report for reference
        if code_metadata:
            metadata_path = os.path.join(error_dir, "isco_code_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(code_metadata, f, indent=2)
        
        if code_hierarchy:
            hierarchy_path = os.path.join(error_dir, "isco_code_hierarchy.json")
            with open(hierarchy_path, "w") as f:
                json.dump(code_hierarchy, f, indent=2)
        
    except Exception as e:
        logger.warning(f"Error performing error analysis: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return metrics

if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Evaluate model
    evaluate_model(config)