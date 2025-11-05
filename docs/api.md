# API Reference

The ISCO Pipeline provides a FastAPI service for online prediction scoring.

## Starting the API

```bash
python api_server.py
```

The server listens on `http://localhost:8000` with interactive documentation at `/docs`.

## Endpoints

### Single Job Prediction

**POST** `/predict/job`

Predict ISCO code for a single job.

#### Request Body

```json
{
  "job_title": "Software Engineer",
  "duties_description": "Design and build backend services"
}
```

#### Response

```json
{
  "job_title": "Software Engineer",
  "duties_description": "Design and build backend services",
  "predicted_code": "2512",
  "predicted_occupation": "Software Developers",
  "confidence": 0.95,
  "confidence_grade": "very_high",
  "is_fallback": false,
  "alternatives": [
    {
      "code": "2511",
      "occupation": "Systems Analysts",
      "confidence": 0.03
    },
    {
      "code": "2519",
      "occupation": "Software and Applications Developers Not Elsewhere Classified",
      "confidence": 0.01
    }
  ]
}
```

#### Example cURL

```bash
curl -X POST http://localhost:8000/predict/job \
  -H "Content-Type: application/json" \
  -d '{"job_title":"Software Engineer","duties_description":"Design and build backend services"}'
```

### Batch Prediction

**POST** `/predict/batch`

Predict ISCO codes for multiple jobs in a single request.

#### Request Body

```json
{
  "jobs": [
    {
      "job_title": "Software Engineer",
      "duties_description": "Design and build backend services"
    },
    {
      "job_title": "Data Scientist",
      "duties_description": "Analyze data and build ML models"
    }
  ]
}
```

#### Response

Returns an array of prediction objects with the same structure as single predictions.

### CSV Upload Prediction

**POST** `/predict/csv`

Upload a CSV file and receive predictions for all rows.

#### Request

- Content-Type: `multipart/form-data`
- Form field: `file` (CSV file)
- Required CSV columns: `job_title`, `duties_description`

#### Response

Returns a CSV file with the original data plus prediction columns:
- `predicted_code`
- `predicted_occupation`
- `confidence`
- `confidence_grade`
- `is_fallback`
- `alternative_1` / `_occupation` / `_confidence` (present when an alternative exists)
- `alternative_2` / `_occupation` / `_confidence`

#### Example cURL

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@jobs.csv" \
  -o predictions.csv
```

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `job_title` | string | Original job title from the request |
| `duties_description` | string | Original job description |
| `predicted_code` | string | Predicted ISCO-08 code (4-digit or truncated to 3-digit on fallback) |
| `predicted_occupation` | string | Human-readable occupation title sourced from the reference data |
| `confidence` | float | Model confidence (0-1) |
| `confidence_grade` | string | Bucketed confidence level (`very_low` … `very_high`) |
| `is_fallback` | boolean | `true` when confidence is below the configured threshold |
| `alternatives` | array | Up to two alternative predictions with `code`, `occupation`, and `confidence` |

## Model Caching

The API uses an LRU cache for predictions, keyed by (job_title, duties_description) pairs. This improves response times for repeated queries.

## Configuration

The API loads the model from the path specified in `config.yaml`:

```yaml
output:
  best_model_dir: "models/best_model"
```

Confidence thresholds and other inference settings are also read from `config.yaml`:

```yaml
model:
  confidence_threshold: 0.1
  max_seq_length: 256
```

## Error Handling

The API returns appropriate HTTP status codes:

- **200** — Success
- **400** — Bad request (invalid input format)
- **422** — Validation error (missing required fields)
- **500** — Internal server error (model loading or prediction failure)

## Interactive Documentation

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI, where you can:

- Explore all available endpoints
- Test requests interactively
- View request/response schemas
- Download OpenAPI specification
