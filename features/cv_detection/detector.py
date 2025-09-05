"""
CV Detection Detector

Clean interface for computer vision-based circle detection.
Uses traditional CV algorithms for precise circle detection.
"""

import cv2
import numpy as np
from .algorithms import segment_circles_and_mask, create_combined_image
from .models import Circle


class CVDetector:
    """
    Computer vision-based circle detector.
    
    This class provides a clean interface for traditional CV-based
    circle detection using OpenCV algorithms.
    """
    
    def __init__(self):
        """
        Initialize the CV detector.
        
        No parameters needed as this uses traditional CV algorithms
        that don't require model loading.
        """
        pass
    
    def detect(self, image):
        """
        Detect circular objects in an image using computer vision.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (mask, circles):
            - mask: Binary mask where 0=background, 255=foreground circles
            - circles: List of detected Circle objects
        """
        return segment_circles_and_mask(image)
    
    def detect_circles_only(self, image):
        """
        Detect circular objects and return only the circles.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected Circle objects
        """
        _, circles = self.detect(image)
        return circles
    
    def detect_mask_only(self, image):
        """
        Detect circular objects and return only the mask.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Binary mask where 0=background, 255=foreground circles
        """
        mask, _ = self.detect(image)
        return mask
    
    def create_visualization(self, image, mask, circles=None):
        """
        Create a combined visualization of the original image and mask.
        
        Args:
            image: Original input image
            mask: Binary mask of detected circles
            circles: List of detected circles (optional)
            
        Returns:
            Combined image with original and mask side by side
        """
        return create_combined_image(image, mask, circles)
    
    def get_detection_summary(self, circles):
        """
        Get a summary of the detection results.
        
        Args:
            circles: List of detected circles
            
        Returns:
            Dictionary with detection summary statistics
        """
        if not circles:
            return {
                "count": 0,
                "total_area": 0.0,
                "average_radius": 0.0,
                "min_radius": 0.0,
                "max_radius": 0.0
            }
        
        radii = [circle.r for circle in circles]
        areas = [circle.area for circle in circles]
        
        return {
            "count": len(circles),
            "total_area": sum(areas),
            "average_radius": sum(radii) / len(radii),
            "min_radius": min(radii),
            "max_radius": max(radii),
            "circles": [
                {
                    "id": i + 1,
                    "center": circle.center,
                    "radius": circle.r,
                    "bbox": circle.bbox,
                    "area": circle.area
                }
                for i, circle in enumerate(circles)
            ]
        }
