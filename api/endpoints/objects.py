"""
Object detection and retrieval endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database import get_db
from models.tables import Image, Detection
from models.schemas import ImageObjectsResponse, ObjectDetailsResponse, ObjectInfo
from services.detection_service import DetectionService
from core.config import settings
import uuid

router = APIRouter()

@router.post("/detect/{image_id}")
def detect_objects(image_id: int, db: Session = Depends(get_db)):
    """
    Detect circular objects in an uploaded image
    """
    # Get image from database
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # Initialize detection service
        detection_service = DetectionService()
        
        # Get full image path
        image_path = settings.storage_dir / image.file_path
        
        # Run detection
        circles = detection_service.detect_image(str(image_path))
        
        # Clear existing detections for this image
        db.query(Detection).filter(Detection.image_id == image_id).delete()
        
        # Save new detections
        for circle, confidence in circles:
            # Generate unique object ID
            object_id = str(uuid.uuid4())
            
            # Calculate bounding box
            bbox_x1 = circle.cx - circle.r
            bbox_y1 = circle.cy - circle.r
            bbox_x2 = circle.cx + circle.r
            bbox_y2 = circle.cy + circle.r
            
            detection = Detection(
                image_id=image_id,
                object_id=object_id,
                bbox_x1=bbox_x1,
                bbox_y1=bbox_y1,
                bbox_x2=bbox_x2,
                bbox_y2=bbox_y2,
                centroid_x=circle.cx,
                centroid_y=circle.cy,
                radius=circle.r,
                confidence=confidence
            )
            
            db.add(detection)
        
        db.commit()
        
        return {"message": f"Detected {len(circles)} objects", "object_count": len(circles)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting objects: {str(e)}")

@router.get("/image/{image_id}", response_model=ImageObjectsResponse)
def get_objects_for_image(image_id: int, db: Session = Depends(get_db)):
    """
    Get list of all circular objects (id and bounding box) for queried image
    """
    # Check if image exists
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get detections for this image
    detections = db.query(Detection).filter(Detection.image_id == image_id).all()
    
    # Convert to response format
    objects = []
    for detection in detections:
        object_info = ObjectInfo(
            object_id=detection.object_id,
            bbox={
                "x1": detection.bbox_x1,
                "y1": detection.bbox_y1,
                "x2": detection.bbox_x2,
                "y2": detection.bbox_y2
            },
            centroid={
                "x": detection.centroid_x,
                "y": detection.centroid_y
            },
            radius=detection.radius,
            confidence=f"{detection.confidence:.2f}"
        )
        objects.append(object_info)
    
    return ImageObjectsResponse(
        image_id=image_id,
        image_filename=image.filename,
        total_objects=len(objects),
        objects=objects
    )

@router.get("/{object_id}", response_model=ObjectDetailsResponse)
def get_object_details(object_id: str, db: Session = Depends(get_db)):
    """
    Find bounding box, centroid and radius for queried circular object
    """
    # Get detection from database
    detection = db.query(Detection).filter(Detection.object_id == object_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Object not found")
    
    # Get image info
    image = db.query(Image).filter(Image.id == detection.image_id).first()
    
    return ObjectDetailsResponse(
        object_id=detection.object_id,
        image_id=detection.image_id,
        image_filename=image.filename if image else "Unknown",
        bbox={
            "x1": detection.bbox_x1,
            "y1": detection.bbox_y1,
            "x2": detection.bbox_x2,
            "y2": detection.bbox_y2
        },
        centroid={
            "x": detection.centroid_x,
            "y": detection.centroid_y
        },
        radius=detection.radius,
        confidence=f"{detection.confidence:.2f}",
        created_at=detection.created_at
    )

@router.get("/", response_model=list[ObjectDetailsResponse])
def list_all_objects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    List all detected objects
    """
    detections = db.query(Detection).offset(skip).limit(limit).all()
    
    objects = []
    for detection in detections:
        image = db.query(Image).filter(Image.id == detection.image_id).first()
        
        objects.append(ObjectDetailsResponse(
            object_id=detection.object_id,
            image_id=detection.image_id,
            image_filename=image.filename if image else "Unknown",
            bbox={
                "x1": detection.bbox_x1,
                "y1": detection.bbox_y1,
                "x2": detection.bbox_x2,
                "y2": detection.bbox_y2
            },
            centroid={
                "x": detection.centroid_x,
                "y": detection.centroid_y
            },
            radius=detection.radius,
            confidence=f"{detection.confidence:.2f}",
            created_at=detection.created_at
        ))
    
    return objects
