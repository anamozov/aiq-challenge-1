"""
File storage utilities
"""

import os
import uuid
from pathlib import Path
from fastapi import UploadFile
from core.config import settings

def save_uploaded_file(file: UploadFile, subdirectory: str = "uploads") -> str:
    """
    Save uploaded file to storage directory
    
    Args:
        file: Uploaded file
        subdirectory: Subdirectory within storage (default: uploads)
    
    Returns:
        str: Relative path to saved file
    """
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Create subdirectory if it doesn't exist
    target_dir = settings.storage_dir / subdirectory
    target_dir.mkdir(exist_ok=True)
    
    # Save file
    file_path = target_dir / unique_filename
    with open(file_path, "wb") as buffer:
        content = file.file.read()
        buffer.write(content)
    
    # Return relative path
    return str(file_path.relative_to(settings.storage_dir))

def get_file_path(relative_path: str) -> Path:
    """
    Get full file path from relative path
    
    Args:
        relative_path: Relative path from storage directory
    
    Returns:
        Path: Full file path
    """
    return settings.storage_dir / relative_path

def delete_file(relative_path: str) -> bool:
    """
    Delete file from storage
    
    Args:
        relative_path: Relative path from storage directory
    
    Returns:
        bool: True if deleted successfully
    """
    try:
        file_path = get_file_path(relative_path)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception:
        return False
