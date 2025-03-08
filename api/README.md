# ISCO Classification API

This API provides endpoints for predicting ISCO-08 occupation codes from job titles and descriptions.

## Quick Start

To start the API server:

```bash
python api_server.py
```

This will start the server on http://localhost:8000

## API Endpoints

The API provides the following endpoints:

### 1. Predict ISCO code for a single job

```
POST /predict/job
```

Request body:
```json
{
  "job_title": "Software Engineer",
  "duties_description": "Designs and develops software applications. Writes and tests code."
}
```

### 2. Predict ISCO codes for a batch of jobs

```
POST /predict/batch
```

Request body:
```json
{
  "jobs": [
    {
      "job_title": "Software Engineer",
      "duties_description": "Designs and develops software applications. Writes and tests code."
    },
    {
      "job_title": "Accountant",
      "duties_description": "Prepares financial reports and maintains accounting records for the company."
    }
  ]
}
```

### 3. Predict ISCO codes from a CSV file

```
POST /predict/csv
```

Accepts a multipart/form-data request with a CSV file containing at least the columns `job_title` and `duties_description`.

Returns a CSV file with the original data plus prediction columns.

### 4. Health check

```
GET /health
```

Returns the health status of the API, including whether the model is loaded successfully.

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc