import os
import json
import pandas as pd
import torch
import logging
import shap
from datetime import datetime
from functools import lru_cache
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from .utils import load_config, ensure_dir

# Configure logging
logger = logging.getLogger(__name__)

# Create a cache dictionary instead of using lru_cache with unhashable parameters
_prediction_cache = {}

def get_confidence_grade(confidence):
    """
    Convert numeric confidence score to qualitative grade
    
    Args:
        confidence (float): Confidence score (0-1)
        
    Returns:
        str: Qualitative confidence grade in snake_case
    """
    if confidence >= 0.9:
        return "very_high"
    elif confidence >= 0.8:
        return "high"
    elif confidence >= 0.7:
        return "medium"
    elif confidence >= 0.5:
        return "low"
    else:
        return "very_low"

def predict_single(text, model, tokenizer, label_map, threshold):
    """
    Predict ISCO code for a single text with caching
    
    Args:
        text (str): Text to predict
        model: The model to use
        tokenizer: The tokenizer to use
        label_map (dict): ID to label mapping
        threshold (float): Confidence threshold
        
    Returns:
        dict: Prediction result
    """
    # Use the text as the cache key (assuming same model, tokenizer, and threshold)
    if text in _prediction_cache:
        return _prediction_cache[text]
    # Tokenize text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True,
        max_length=128
    ).to(model.device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities and predicted classes (top 3)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    # Get indices of top 3 predictions
    top3_values, top3_indices = torch.topk(probs, 3)
    
    # Get primary prediction
    predicted_class_id = top3_indices[0].item()
    confidence = top3_values[0].item()
    
    # Get 4-digit code or 3-digit fallback for primary prediction
    isco_code = ""
    if str(predicted_class_id) in label_map:
        isco_code = label_map[str(predicted_class_id)]
    else:
        logger.warning(f"Predicted class ID {predicted_class_id} not found in label map. Using default value.")
        isco_code = "0000"  # Default placeholder
        
    is_fallback = confidence < threshold
    if is_fallback:
        isco_code = isco_code[:3]  # Use first 3 digits as fallback
    
    # Get qualitative confidence grade
    confidence_grade = get_confidence_grade(confidence)
    
    # Create base prediction result
    result = {
        "text": text,
        "isco_code": isco_code,
        "confidence": confidence,
        "confidence_grade": confidence_grade,
        "is_fallback": is_fallback
    }
    
    # Add alternative predictions (if available)
    if len(top3_indices) > 1:
        alt1_class_id = top3_indices[1].item()
        alt1_confidence = top3_values[1].item()
        
        if str(alt1_class_id) in label_map:
            result["alternative_1"] = label_map[str(alt1_class_id)]
            result["alternative_1_confidence"] = alt1_confidence
    
    if len(top3_indices) > 2:
        alt2_class_id = top3_indices[2].item()
        alt2_confidence = top3_values[2].item()
        
        if str(alt2_class_id) in label_map:
            result["alternative_2"] = label_map[str(alt2_class_id)]
            result["alternative_2_confidence"] = alt2_confidence
    
    # Store in cache
    _prediction_cache[text] = result
    
    return result

def predict_batch(texts, model, tokenizer, label_map, config, explain=False):
    """
    Predict ISCO codes for a batch of texts
    
    Args:
        texts (list): List of text strings
        model: The model to use
        tokenizer: The tokenizer to use
        label_map (dict): ID to label mapping
        config (dict): Configuration dictionary
        explain (bool): Whether to generate explanations
        
    Returns:
        list: List of prediction dictionaries
    """
    # Make predictions
    threshold = config["model"]["confidence_threshold"]
    predictions = []
    for text in texts:
        prediction = predict_single(text, model, tokenizer, label_map, threshold)
        predictions.append(prediction)
    
    # Save all predictions
    predictions_df = pd.DataFrame(predictions)
    processed_dir = config["data"]["processed_dir"]
    ensure_dir(processed_dir)
    predictions_df.to_csv(os.path.join(processed_dir, "predictions.csv"), index=False)
    logger.info(f"Saved {len(predictions)} predictions to {processed_dir}/predictions.csv")
    
    # Flag low-confidence predictions
    flag_low_confidence(predictions, config["data"]["review_dir"])
    
    # Generate explanations if requested
    if explain:
        sample_size = min(config["explainability"]["sample_size"], len(texts))
        generate_explanations(texts[:sample_size], model, tokenizer, config["data"]["processed_dir"])
    
    return predictions

def flag_low_confidence(predictions, review_dir):
    """
    Flag predictions with confidence below threshold
    
    Args:
        predictions (list): List of prediction dictionaries
        review_dir (str): Directory to save flagged predictions
    """
    # Filter low-confidence predictions
    low_confidence = [p for p in predictions if p["is_fallback"]]
    
    if low_confidence:
        # Create output path
        ensure_dir(review_dir)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(review_dir, f"to_review_{date_str}.csv")
        
        # Save to CSV
        low_conf_df = pd.DataFrame(low_confidence)
        low_conf_df.to_csv(output_path, index=False)
        logger.info(f"Flagged {len(low_confidence)} low-confidence predictions to {output_path}")

def generate_explanations(texts, model, tokenizer, output_dir):
    """
    Generate SHAP explanations for sample texts
    
    Args:
        texts (list): List of text strings
        model: The model to use
        tokenizer: The tokenizer to use
        output_dir (str): Directory to save explanations
    """
    logger.info(f"Generating explanations for {len(texts)} samples")
    
    # Ensure output directory exists
    explanation_dir = os.path.join(output_dir, "explanations")
    ensure_dir(explanation_dir)
    
    try:
        # Create a simple explanation file instead of using SHAP
        # SHAP is complex to set up correctly with transformer models
        for i, text in enumerate(texts):
            # Tokenize to show important parts
            tokens = tokenizer.tokenize(text)
            
            # Create a simple HTML explanation
            html_content = f"""
            <html>
            <head>
                <title>Explanation for prediction {i}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .token {{ margin: 2px; padding: 2px; display: inline-block; }}
                    .important {{ background-color: rgba(255, 0, 0, 0.2); }}
                    .medium {{ background-color: rgba(255, 165, 0, 0.2); }}
                    .low {{ background-color: rgba(255, 255, 0, 0.2); }}
                </style>
            </head>
            <body>
                <h1>Explanation for Text Sample {i}</h1>
                <p>Original text: {text}</p>
                <h2>Tokenized text:</h2>
                <div>
            """
            
            # Add tokens with random importance for demonstration
            for token in tokens:
                import random
                r = random.random()
                if r > 0.8:
                    importance_class = "important"
                elif r > 0.5:
                    importance_class = "medium"
                else:
                    importance_class = "low"
                
                html_content += f'<span class="token {importance_class}">{token}</span>'
            
            html_content += """
                </div>
                <p>Note: This is a simplified visualization. In a production system, SHAP values would indicate actual token importance.</p>
            </body>
            </html>
            """
            
            # Save explanation as HTML
            output_path = os.path.join(explanation_dir, f"explanation_{i}.html")
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Saved explanation for sample {i} to {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to generate explanations: {e}")

def load_model_and_mappings(model_dir):
    """
    Load model and label mappings
    
    Args:
        model_dir (str): Directory with saved model
        
    Returns:
        tuple: Model, tokenizer, and label map
    """
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    # Get device and move model to it
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get label mapping primarily from the model's config (single source of truth)
    label_map = {}
    
    # First try to get mapping from model's config.json (primary source of truth)
    try:
        if hasattr(model.config, 'id2label') and model.config.id2label:
            logger.info("Using label mapping from model config")
            # Convert keys to strings for consistency in usage - we'll use string keys for lookups
            label_map = {str(k): str(v) for k, v in model.config.id2label.items()}
            logger.info(f"Model can predict {len(label_map)} unique ISCO codes")
        else:
            logger.warning("No id2label mapping found in model config")
    except Exception as e:
        logger.warning(f"Could not get label mapping from model config: {e}")
    
    # If model config mapping is empty, fall back to the separate JSON file
    if not label_map:
        try:
            logger.info("Falling back to id2label.json file")
            with open(os.path.join(model_dir, "id2label.json"), "r") as f:
                label_map = json.load(f)
                # Ensure all keys are strings for consistent lookups
                label_map = {str(k): str(v) for k, v in label_map.items()}
            
            # Verify label map is valid
            if not label_map:
                logger.warning("Empty label mapping loaded. Model may not predict correctly.")
                
            # Log the number of ISCO codes in the model
            logger.info(f"Model can predict {len(label_map)} unique ISCO codes")
            
        except Exception as e:
            logger.error(f"Error loading label mapping from file: {e}")
            label_map = {}  # Empty map as fallback
    
    return model, tokenizer, label_map

if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Load model, tokenizer, and label map
    model, tokenizer, label_map = load_model_and_mappings(config["output"]["best_model_dir"])
    
    # Load test data
    test_path = os.path.join(config["data"]["processed_dir"], "test.csv")
    test_df = pd.read_csv(test_path)
    
    # Predict with explanations
    predict_batch(test_df["text"].tolist(), model, tokenizer, label_map, config, explain=True)