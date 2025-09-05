#!/bin/bash
set -e

echo "Starting Circular Object Detection API..."

# Create necessary directories with proper permissions
mkdir -p storage/uploads storage/results storage/models features/yolov11/weights
chmod 755 storage/uploads storage/results storage/models features/yolov11/weights

# Wait for storage directory to be ready
echo "Waiting for storage directory to be ready..."
while [ ! -d "storage" ]; do
    echo "Storage directory not ready, waiting..."
    sleep 2
done

# Initialize database if it doesn't exist or tables are missing
echo "Checking database status..."

# Check if database file exists
if [ ! -f "storage/detection.db" ]; then
    echo "Database file not found, initializing..."
    python scripts/setup_database.py
    echo "Database initialization completed!"
else
    # Check if tables exist
    echo "Database file exists, checking tables..."
    if ! python -c "
import sqlite3
import sys
try:
    conn = sqlite3.connect('storage/detection.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\" AND name=\"images\"')
    result = cursor.fetchone()
    conn.close()
    if not result:
        print('Tables missing')
        sys.exit(1)
    else:
        print('Tables exist')
        sys.exit(0)
except Exception as e:
    print(f'Error checking tables: {e}')
    sys.exit(1)
" 2>/dev/null; then
        echo "Database tables are missing, initializing..."
        python scripts/setup_database.py
        echo "Database initialization completed!"
    else
        echo "Database already initialized and tables exist."
    fi
fi

# Check if YOLO model exists
if [ ! -f "features/yolov11/weights/best.pt" ]; then
    echo "Warning: YOLO model not found at features/yolov11/weights/best.pt"
    echo "Please ensure the model is trained and available"
fi

# Start the application
echo "Starting FastAPI server..."
exec python main.py
