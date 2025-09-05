#!/bin/bash
set -e

echo "Starting Circular Object Detection API..."

# Create necessary directories
mkdir -p storage/uploads storage/results storage/models

# Initialize database if it doesn't exist
if [ ! -f "detection.db" ]; then
    echo "Initializing database..."
    python scripts/setup_database.py
fi

# Check if YOLO model exists
if [ ! -f "features/yolov11/weights/best.pt" ]; then
    echo "Warning: YOLO model not found at features/yolov11/weights/best.pt"
    echo "Please ensure the model is trained and available"
fi

# Start the application
echo "Starting FastAPI server..."
exec python main.py
