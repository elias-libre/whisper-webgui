#!/bin/bash

echo "Starting Whisper WebGUI..."
echo "Open http://localhost:8000 in your browser"

./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
