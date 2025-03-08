from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict
import pandas as pd
import tempfile
import io
import os
import csv
import json
from datetime import datetime

from ..dependencies import get_model_and_tokenizer
from ..models import (
    JobInput, 
    JobBatchInput,
    ISCOPrediction, 
    BatchPredictionResponse
)
from ..prediction_service import predict_single_job, predict_batch_jobs

router = APIRouter(
    prefix="/predict",
    tags=["prediction"],
    responses={404: {"description": "Not found"}},
)


@router.post("/job", response_model=ISCOPrediction)
async def predict_job(
    job: JobInput,
    model_data=Depends(get_model_and_tokenizer)
):
    """
    Predict ISCO code for a single job.
    """
    model, tokenizer, label_map, reference_data = model_data
    
    prediction = predict_single_job(
        job=job,
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        reference_data=reference_data
    )
    
    return prediction


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch: JobBatchInput,
    model_data=Depends(get_model_and_tokenizer)
):
    """
    Predict ISCO codes for a batch of jobs.
    """
    model, tokenizer, label_map, reference_data = model_data
    
    predictions = predict_batch_jobs(
        jobs=batch.jobs,
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        reference_data=reference_data
    )
    
    return BatchPredictionResponse(predictions=predictions)


@router.post("/csv", response_class=FileResponse)
async def predict_from_csv(
    file: UploadFile = File(...),
    model_data=Depends(get_model_and_tokenizer)
):
    """
    Predict ISCO codes from a CSV file.
    The CSV file must have columns 'job_title' and 'duties_description'.
    Returns a CSV file with predictions added.
    """
    model, tokenizer, label_map, reference_data = model_data
    
    # Check file format
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a CSV file."
        )
    
    # Read CSV file
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}"
        )
    
    # Check required columns
    required_columns = ['job_title', 'duties_description']
    if not all(col in df.columns for col in required_columns):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV file must contain columns: {', '.join(required_columns)}"
        )
    
    # Convert dataframe to list of JobInput objects
    jobs = [
        JobInput(job_title=row['job_title'], duties_description=row['duties_description'])
        for _, row in df.iterrows()
    ]
    
    # Get predictions
    predictions = predict_batch_jobs(
        jobs=jobs,
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        reference_data=reference_data
    )
    
    # Add predictions to dataframe
    for i, pred in enumerate(predictions):
        df.loc[i, 'predicted_code'] = pred.predicted_code
        df.loc[i, 'predicted_occupation'] = pred.predicted_occupation
        df.loc[i, 'confidence'] = pred.confidence
        df.loc[i, 'confidence_grade'] = pred.confidence_grade
        df.loc[i, 'is_fallback'] = pred.is_fallback
        
        # Add alternatives
        if len(pred.alternatives) > 0:
            df.loc[i, 'alternative_1'] = pred.alternatives[0].code
            df.loc[i, 'alternative_1_occupation'] = pred.alternatives[0].occupation
            df.loc[i, 'alternative_1_confidence'] = pred.alternatives[0].confidence
            
        if len(pred.alternatives) > 1:
            df.loc[i, 'alternative_2'] = pred.alternatives[1].code
            df.loc[i, 'alternative_2_occupation'] = pred.alternatives[1].occupation
            df.loc[i, 'alternative_2_confidence'] = pred.alternatives[1].confidence
    
    # Create temporary file for response
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"predictions_{timestamp}.csv"
    
    # Use a directory that will be accessible
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, output_filename)
    
    # Save predictions to CSV
    df.to_csv(output_path, index=False)
    
    # Return CSV file as response
    return FileResponse(
        path=output_path,
        filename=output_filename,
        media_type='text/csv'
    )