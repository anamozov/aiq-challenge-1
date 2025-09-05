"""
Object detection and retrieval endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from models.database import get_db
from models.tables import Image, Detection
from models.schemas import ImageObjectsResponse, ObjectDetailsResponse, ObjectInfo
from services.detection_service import DetectionService
from core.config import settings
import uuid
import os
from pathlib import Path

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
        yolo_detections, circles_with_confidence = detection_service.detect_image(str(image_path))
        
        # Clear existing detections for this image
        db.query(Detection).filter(Detection.image_id == image_id).delete()
        
        # Save new detections - use YOLO bounding boxes but circle radius
        for i, (circle, confidence) in enumerate(circles_with_confidence):
            # Generate unique object ID
            object_id = str(uuid.uuid4())
            
            # Use YOLO bounding box coordinates
            yolo_bbox = yolo_detections[i]  # (x1, y1, x2, y2, conf)
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, yolo_conf = yolo_bbox
            
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
        
        # Auto-save visual results if objects were detected
        visual_files = {}
        if circles_with_confidence:
            try:
                # Extract circles from tuples
                circle_objects = []
                for circle, confidence in circles_with_confidence:
                    circle_objects.append(circle)
                
                # Save visual results
                results_dir = str(settings.storage_dir / "results")
                visual_files = detection_service.save_detection_results(str(image_path), yolo_detections, circle_objects, results_dir)
                
            except Exception as e:
                print(f"Warning: Could not save visual results: {e}")
                # Don't fail the detection if visual saving fails
        
        response = {
            "message": f"Detected {len(circles_with_confidence)} objects", 
            "object_count": len(circles_with_confidence)
        }
        
        if visual_files:
            response["visual_results"] = {
                "message": "Visual results saved automatically",
                "files": visual_files
            }
        
        return response
        
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

@router.get("/visual/{image_id}")
def get_visual_results(image_id: int, result_type: str = "combined", db: Session = Depends(get_db)):
    """
    Generate and return visual detection results for an image
    
    Args:
        image_id: ID of the image
        result_type: Type of visual result ("detection", "mask", "combined")
    
    Returns:
        Image file with detection results
    """
    # Get image from database
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get detections for this image
    detections = db.query(Detection).filter(Detection.image_id == image_id).all()
    if not detections:
        raise HTTPException(status_code=404, detail="No detections found for this image. Run detection first.")
    
    try:
        # Initialize detection service
        detection_service = DetectionService()
        
        # Get full image path
        image_path = settings.storage_dir / image.file_path
        
        # Convert detections to circles format
        from features.cv_detection.models import Circle
        circles = []
        for detection in detections:
            circle = Circle(
                cx=detection.centroid_x,
                cy=detection.centroid_y,
                r=detection.radius
            )
            circles.append(circle)
        
        # Load original image
        import cv2
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise HTTPException(status_code=500, detail="Could not load original image")
        
        # Get YOLO detections from database (we need to reconstruct them)
        # For now, we'll use the stored bounding boxes as YOLO detections
        yolo_detections = []
        for detection in detections:
            yolo_detections.append((detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2, detection.confidence))
        
        # Generate the requested visual result
        if result_type == "detection":
            result_image = detection_service.draw_yolo_boxes(original_image, yolo_detections)
        elif result_type == "mask":
            mask = detection_service.create_mask(original_image.shape, circles)
            result_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif result_type == "combined":
            mask = detection_service.create_mask(original_image.shape, circles)
            result_image = detection_service.create_combined_image(original_image, mask, yolo_detections, circles)
        else:
            raise HTTPException(status_code=400, detail="Invalid result_type. Use 'detection', 'mask', or 'combined'")
        
        # Save temporary result file
        results_dir = settings.storage_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        temp_filename = f"{image_id}_{result_type}_temp.jpg"
        temp_path = results_dir / temp_filename
        cv2.imwrite(str(temp_path), result_image)
        
        # Return the file
        return FileResponse(
            path=str(temp_path),
            media_type="image/jpeg",
            filename=f"image_{image_id}_{result_type}.jpg"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visual results: {str(e)}")

@router.get("/visual/{image_id}/save")
def save_visual_results(image_id: int, db: Session = Depends(get_db)):
    """
    Generate and save all visual detection results for an image to storage/results/
    
    Args:
        image_id: ID of the image
    
    Returns:
        Dictionary with paths to saved files
    """
    # Get image from database
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get detections for this image
    detections = db.query(Detection).filter(Detection.image_id == image_id).all()
    if not detections:
        raise HTTPException(status_code=404, detail="No detections found for this image. Run detection first.")
    
    try:
        # Initialize detection service
        detection_service = DetectionService()
        
        # Get full image path
        image_path = settings.storage_dir / image.file_path
        
        # Convert detections to circles format
        from features.cv_detection.models import Circle
        circles = []
        for detection in detections:
            circle = Circle(
                cx=detection.centroid_x,
                cy=detection.centroid_y,
                r=detection.radius
            )
            circles.append(circle)
        
        # Get YOLO detections from database
        yolo_detections = []
        for detection in detections:
            yolo_detections.append((detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2, detection.confidence))
        
        # Save detection results
        results_dir = str(settings.storage_dir / "results")
        result_files = detection_service.save_detection_results(str(image_path), yolo_detections, circles, results_dir)
        
        return {
            "message": "Visual results saved successfully",
            "image_id": image_id,
            "files": result_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving visual results: {str(e)}")
