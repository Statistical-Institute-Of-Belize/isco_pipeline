from typing import Dict, List, Any

import pandas as pd

# Import from the original predict.py to ensure consistency
from src.predict import predict_single
from src.preprocess import clean_and_combine_text

from .models import (
    JobInput, 
    ISCOPrediction, 
    AlternativePrediction
)
from .config import settings


def _format_job_text(job_title: str, duties_description: str) -> str:
    """Format inputs using the same preprocessing as training."""
    titles = pd.Series([job_title])
    descriptions = pd.Series([duties_description])
    return clean_and_combine_text(titles, descriptions).iloc[0]


def predict_single_job(
    job: JobInput,
    model: Any, 
    tokenizer: Any, 
    label_map: Dict[int, str],
    reference_data: Dict[str, str],
    threshold: float = None
) -> ISCOPrediction:
    """
    Predict ISCO code for a single job.
    
    Args:
        job: JobInput object with job title and description
        model: Pre-loaded RoBERTa model
        tokenizer: Pre-loaded RoBERTa tokenizer
        label_map: Dictionary mapping from class IDs to ISCO codes
        reference_data: Dictionary mapping from ISCO codes to occupation titles
        threshold: Confidence threshold for predictions
    
    Returns:
        ISCOPrediction object
    """
    if threshold is None:
        threshold = settings.CONFIDENCE_THRESHOLD
    
    # Combine job title and description - same format as in the CLI pipeline
    text = _format_job_text(job.job_title, job.duties_description)
    
    # Use the same prediction function as the CLI pipeline for consistency
    result = predict_single(
        text,
        model,
        tokenizer,
        label_map,
        threshold,
        max_length=settings.MAX_SEQ_LENGTH,
    )
    
    # Create alternatives list
    alternatives = []
    
    # Add alternative 1 if available
    if "alternative_1" in result:
        alt1_code = result["alternative_1"]
        alt1_title = reference_data.get(alt1_code)
        alt1_confidence = result["alternative_1_confidence"]
        
        alternatives.append(
            AlternativePrediction(
                code=alt1_code,
                occupation=alt1_title,
                confidence=alt1_confidence
            )
        )
    
    # Add alternative 2 if available  
    if "alternative_2" in result:
        alt2_code = result["alternative_2"]
        alt2_title = reference_data.get(alt2_code)
        alt2_confidence = result["alternative_2_confidence"]
        
        alternatives.append(
            AlternativePrediction(
                code=alt2_code,
                occupation=alt2_title,
                confidence=alt2_confidence
            )
        )
    
    # Create prediction object
    prediction = ISCOPrediction(
        job_title=job.job_title,
        duties_description=job.duties_description,
        predicted_code=result["isco_code"],
        predicted_occupation=reference_data.get(result["isco_code"]),
        confidence=result["confidence"],
        confidence_grade=result["confidence_grade"],
        is_fallback=result["is_fallback"],
        alternatives=alternatives
    )
    
    return prediction


def predict_batch_jobs(
    jobs: List[JobInput],
    model: Any, 
    tokenizer: Any, 
    label_map: Dict[int, str],
    reference_data: Dict[str, str],
    threshold: float = None
) -> List[ISCOPrediction]:
    """
    Predict ISCO codes for a batch of jobs.
    
    Args:
        jobs: List of JobInput objects
        model: Pre-loaded RoBERTa model
        tokenizer: Pre-loaded RoBERTa tokenizer
        label_map: Dictionary mapping from class IDs to ISCO codes
        reference_data: Dictionary mapping from ISCO codes to occupation titles
        threshold: Confidence threshold for predictions
    
    Returns:
        List of ISCOPrediction objects
    """
    if threshold is None:
        threshold = settings.CONFIDENCE_THRESHOLD
        
    # Process each job individually
    predictions = []
    for job in jobs:
        prediction = predict_single_job(
            job=job,
            model=model,
            tokenizer=tokenizer,
            label_map=label_map,
            reference_data=reference_data,
            threshold=threshold
        )
        predictions.append(prediction)
    
    return predictions
