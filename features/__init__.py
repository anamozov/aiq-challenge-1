"""
Features package 
This package contains the main detection features:
- yolov11: detection using YOLOv11
- cv_detection: mask generation using opencv 
"""

from .yolov11.detector import YOLOv11Detector
from .cv_detection.detector import CVDetector

__all__ = ['YOLOv11Detector', 'CVDetector']
