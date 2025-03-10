#!/bin/bash
set -e

# Create folders if they don't exist
mkdir -p data/processed data/raw data/reference logs models/best_model

if [ "$1" = "api" ] || [ -z "$1" ]; then
    echo "Starting API server..."
    exec python api_server.py
elif [ "$1" = "cli" ]; then
    # Shift the first argument so that $@ contains all CLI arguments
    shift
    echo "Running CLI command with arguments: $@"
    exec python main.py "$@"
else
    # If first argument looks like a flag, assume API mode
    if [ "${1:0:1}" = "-" ]; then
        echo "Starting API server with arguments: $@"
        exec python api_server.py "$@"
    else
        echo "Running command: $@"
        exec "$@"
    fi
fi