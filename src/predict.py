import os
import json
import pandas as pd
import torch
import logging
import numpy as np
from datetime import datetime
from functools import lru_cache
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import html

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
        max_len = config["model"].get("max_seq_length", 160) if "model" in config else 160
        generate_explanations(
            texts[:sample_size],
            model,
            tokenizer,
            config["data"]["processed_dir"],
            max_length=max_len
        )
    
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

def generate_explanations(texts, model, tokenizer, output_dir, max_length=160):
    """
    Generate attention-based explanations for sample texts.

    Args:
        texts (list): List of text strings
        model: The model to use
        tokenizer: The tokenizer to use
        output_dir (str): Directory to save explanations
    """
    logger.info(f"Generating attention-based explanations for {len(texts)} samples")

    explanation_dir = os.path.join(output_dir, "explanations")
    ensure_dir(explanation_dir)

    was_training = model.training
    model.eval()

    try:
        for i, text in enumerate(texts):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            attentions = outputs.attentions
            if not attentions:
                logger.warning("Model did not return attentions; skipping explanation for sample %d", i)
                continue

            attention_tensor = torch.stack(attentions).mean(dim=(0, 2))  # average layers & heads
            cls_attention = attention_tensor[0, 0]

            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                logger.warning("Missing attention mask; skipping explanation for sample %d", i)
                continue

            valid_positions = attention_mask[0].nonzero(as_tuple=True)[0].tolist()
            if len(valid_positions) <= 2:
                logger.warning("Not enough tokens for explanation for sample %d", i)
                continue

            token_indices = valid_positions[1:]  # skip CLS
            token_scores = cls_attention[token_indices].detach().cpu().numpy()
            token_ids = inputs["input_ids"][0, token_indices].detach().cpu().tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            # Filter out trailing separator token
            if tokens and tokens[-1] in {tokenizer.sep_token, "</s>"}:
                tokens = tokens[:-1]
                token_scores = token_scores[:-1]

            if not len(tokens):
                logger.warning("No tokens available after filtering for sample %d", i)
                continue

            scores = token_scores - token_scores.min()
            if np.max(scores) > 0:
                scores = scores / (np.max(scores) + 1e-8)
            else:
                scores = np.zeros_like(scores)

            def score_to_class(value: float) -> str:
                if value >= 0.66:
                    return "high"
                if value >= 0.33:
                    return "medium"
                return "low"

            html_content = [
                "<html>",
                "<head>",
                f"<title>Explanation for prediction {i}</title>",
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 20px; }",
                ".token { margin: 2px; padding: 2px 4px; display: inline-block; border-radius: 4px; white-space: pre; }",
                ".high { background-color: rgba(255, 0, 0, 0.25); }",
                ".medium { background-color: rgba(255, 165, 0, 0.25); }",
                ".low { background-color: rgba(255, 255, 0, 0.2); }",
                ".score { font-size: 0.8em; color: #555; margin-left: 4px; }",
                "</style>",
                "</head>",
                "<body>",
                f"<h1>Attention overview for text sample {i}</h1>",
                f"<p>Original text: {html.escape(text)}</p>",
                "<h2>Token attention (CLS-based heuristic)</h2>",
                "<div>"
            ]

            for token, score in zip(tokens, scores):
                token_class = score_to_class(float(score))
                pretty_token = tokenizer.convert_tokens_to_string([token])
                if not pretty_token:
                    pretty_token = token.replace("Ġ", " ")
                display_token = html.escape(pretty_token)
                if display_token.strip() == "":
                    display_token = "&nbsp;"
                html_content.append(
                    f'<span class="token {token_class}">{display_token}<span class="score">{score:.2f}</span></span>'
                )

            html_content.extend([
                "</div>",
                "<p>Scores reflect averaged CLS attention weights as a heuristic and are not SHAP values.</p>",
                "</body>",
                "</html>"
            ])

            output_path = os.path.join(explanation_dir, f"explanation_{i}.html")
            with open(output_path, "w") as f:
                f.write("\n".join(html_content))

            logger.info(f"Saved attention explanation for sample {i} to {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate explanations: {e}")

    finally:
        model.train(was_training)

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
