#!/bin/bash
set -e

echo "Building Docker image for Circular Object Detection API..."

# Build the Docker image
docker build -t aiq-challenge-1:latest .

echo "Docker image built successfully!"
echo "Image name: aiq-challenge-1:latest"
echo ""
echo "To run the container:"
echo "  docker run -p 8000:8000 -v \$(pwd)/storage:/app/storage aiq-challenge-1:latest"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up"
