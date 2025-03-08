from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union


class JobInput(BaseModel):
    """Single job input for ISCO classification."""
    job_title: str = Field(..., description="Job title to classify", min_length=1)
    duties_description: str = Field(..., description="Description of job duties", min_length=1)


class JobBatchInput(BaseModel):
    """Batch of jobs for ISCO classification."""
    jobs: List[JobInput] = Field(..., description="List of jobs to classify", min_items=1)


class AlternativePrediction(BaseModel):
    """Alternative prediction with code, occupation and confidence."""
    code: str = Field(..., description="ISCO-08 code")
    occupation: Optional[str] = Field(None, description="Occupation title from reference data")
    confidence: float = Field(..., description="Confidence score (0-1)")


class ISCOPrediction(BaseModel):
    """ISCO prediction output."""
    job_title: str = Field(..., description="Original job title")
    duties_description: str = Field(..., description="Original job description")
    predicted_code: str = Field(..., description="Predicted ISCO-08 code")
    predicted_occupation: Optional[str] = Field(None, description="Occupation title from reference data")
    confidence: float = Field(..., description="Confidence score (0-1)")
    confidence_grade: str = Field(..., description="Confidence grade (very_low, low, medium, high, very_high)")
    is_fallback: bool = Field(False, description="Whether this is a fallback to 3-digit code due to low confidence")
    alternatives: List[AlternativePrediction] = Field(
        default_factory=list, 
        description="Alternative predictions (typically the next best 2 predictions)"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[ISCOPrediction]