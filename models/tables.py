"""
SQLAlchemy database models
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from models.database import Base
from datetime import datetime

class Image(Base):
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    detections = relationship("Detection", back_populates="image", cascade="all, delete-orphan")

class Detection(Base):
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    object_id = Column(String, nullable=False, unique=True)  # Unique reference identifier
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    centroid_x = Column(Float, nullable=False)
    centroid_y = Column(Float, nullable=False)
    radius = Column(Float, nullable=False)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    image = relationship("Image", back_populates="detections")

class Evaluation(Base):
    __tablename__ = "evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    evaluation_type = Column(String, nullable=False)  # e.g., "precision", "recall", "f1"
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    details = Column(Text)  # JSON string with additional details
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    image = relationship("Image")
