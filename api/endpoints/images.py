"""
Image upload and management endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from models.database import get_db
from models.tables import Image
from models.schemas import Image as ImageSchema, UploadResponse, ErrorResponse
from core.storage import save_uploaded_file
from core.config import settings
import cv2
import os
from pathlib import Path

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload an image and store it in persistent storage
    """
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {list(settings.allowed_extensions)}"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
            )
        
        # Save file
        file_path = save_uploaded_file(file, "uploads")
        
        # Get image dimensions
        full_path = settings.storage_dir / file_path
        image = cv2.imread(str(full_path))
        if image is None:
            # Clean up saved file
            os.remove(full_path)
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        height, width = image.shape[:2]
        
        # Save to database
        db_image = Image(
            filename=Path(file_path).name,
            original_filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            width=width,
            height=height
        )
        
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        
        return UploadResponse(
            image_id=db_image.id,
            filename=db_image.filename,
            message="Image uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")

@router.get("/{image_id}", response_model=ImageSchema)
def get_image(image_id: int, db: Session = Depends(get_db)):
    """
    Get image information by ID
    """
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return image

@router.get("/", response_model=list[ImageSchema])
def list_images(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    List all uploaded images
    """
    images = db.query(Image).offset(skip).limit(limit).all()
    return images
