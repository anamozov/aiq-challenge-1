"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict

# Base schemas
class ImageBase(BaseModel):
    filename: str
    original_filename: str
    file_path: str
    file_size: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

class ImageCreate(ImageBase):
    pass

class Image(ImageBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class DetectionBase(BaseModel):
    object_id: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    centroid_x: float
    centroid_y: float
    radius: float
    confidence: Optional[float] = None

class DetectionCreate(DetectionBase):
    image_id: int

class Detection(DetectionBase):
    id: int
    image_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class ObjectInfo(BaseModel):
    object_id: str
    bbox: dict  # {"x1": float, "y1": float, "x2": float, "y2": float}
    centroid: dict  # {"x": float, "y": float}
    radius: float
    confidence: Optional[float] = None

class ImageObjectsResponse(BaseModel):
    image_id: int
    image_filename: str
    total_objects: int
    objects: List[ObjectInfo]

class ObjectDetailsResponse(BaseModel):
    object_id: str
    image_id: int
    image_filename: str
    bbox: dict
    centroid: dict
    radius: float
    confidence: Optional[float] = None
    created_at: datetime

class EvaluationBase(BaseModel):
    evaluation_type: str
    metric_name: str
    metric_value: float
    details: Optional[str] = None

class EvaluationCreate(EvaluationBase):
    image_id: int

class Evaluation(EvaluationBase):
    id: int
    image_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# API Response schemas
class UploadResponse(BaseModel):
    image_id: int
    filename: str
    message: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Evaluation schemas
class EvaluationResponse(BaseModel):
    success: bool
    message: str
    metrics: Dict[str, any]  # Allow mixed types for metrics
    model_info: Dict[str, str]  # Changed from any to str for compatibility
    evaluation_time: Optional[str] = None
    
    class Config:
        protected_namespaces = ()  # Allow model_info field
        arbitrary_types_allowed = True  # Allow any types

class EvaluationRequest(BaseModel):
    input_size: Optional[int] = 640
    batch_size: Optional[int] = 4
    save_plots: Optional[bool] = False
