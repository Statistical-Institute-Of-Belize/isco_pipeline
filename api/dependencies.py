import os
import sys
from typing import Dict, Tuple, Any, Optional
from fastapi import HTTPException, status

# Configure paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import from the original modules to maintain consistency
from src.predict import load_model_and_mappings, get_confidence_grade
from src.utils import load_isco_reference
from .config import settings

# Keep a global reference to model data
_model_data = None


def get_model_and_tokenizer() -> Tuple[Any, Any, Dict, Dict]:
    """
    Load the ISCO classification model, tokenizer and mappings.
    Uses caching to avoid reloading the model on each request.
    Uses the original pipeline's load_model_and_mappings function
    for consistency.
    
    Returns:
        tuple: (model, tokenizer, label_map, reference_data)
    """
    global _model_data
    
    # If model is already loaded, return from cache
    if _model_data is not None:
        return _model_data
    
    try:
        # Use the same model loading function as the main pipeline
        model, tokenizer, id2label = load_model_and_mappings(settings.MODEL_PATH)
        
        # Load the reference data using the same function as main pipeline
        reference_data = load_isco_reference(settings.REFERENCE_FILE)
        
        # Cache the model, tokenizer and mappings
        _model_data = (model, tokenizer, id2label, reference_data)
        
        return _model_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )