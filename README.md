# ISCO Classification Pipeline

A machine learning pipeline for automated classification of job descriptions according to the International Standard Classification of Occupations (ISCO-08).

## Overview

This pipeline processes job descriptions (title + duties) and assigns the appropriate ISCO-08 code, providing occupation classification for labor market analysis.

## Documentation

For detailed information, see the docs folder:

- [User Manual](docs/USER_MANUAL.md)
- [Pipeline Diagrams](docs/pipeline_diagrams.md)
- [General Plan](docs/general_plan.md)
- [Coding Instructions](docs/coding_instructions.md)
- [Linux Deployment](docs/linux_deployment.md)

## Project Structure

- `src/`: Core pipeline code (preprocessing, modeling, prediction)
- `api/`: FastAPI server for prediction services
- `models/`: Directory for saved models
- `data/`: Raw and processed data
- `docs/`: Documentation files

## Quick Start

### CLI Usage

```
python main.py --config config.yaml
```

### API Usage

```
python api_server.py
```

Then access the API at `http://localhost:8000/docs`

## Docker Deployment

pending

## License

Copyright © 2025
