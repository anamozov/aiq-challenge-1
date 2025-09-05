"""
config for the AIQ Challenge 1 application
# Model settings maybe later use config from .env

"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Database settings
DATABASE_URL = "sqlite:///./storage/detection.db"

# Storage settings
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
RESULTS_DIR = STORAGE_DIR / "results"
MODELS_DIR = STORAGE_DIR / "models"

for directory in [STORAGE_DIR, UPLOADS_DIR, RESULTS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)
# Model settings 

YOLO_MODEL_PATH = "features/yolov11/weights/best.pt"
YOLO_INPUT_SIZE = 640

# API settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

class Settings:
    database_url = DATABASE_URL
    storage_dir = STORAGE_DIR
    uploads_dir = UPLOADS_DIR
    results_dir = RESULTS_DIR
    models_dir = MODELS_DIR
    yolo_model_path = YOLO_MODEL_PATH
    yolo_input_size = YOLO_INPUT_SIZE
    max_file_size = MAX_FILE_SIZE
    allowed_extensions = ALLOWED_EXTENSIONS

settings = Settings()
