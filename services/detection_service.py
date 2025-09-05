"""
Detection service for AIQ Challenge 1 -  detection logic migrated from hybrid_detection.py
"""

import cv2
import numpy as np
from pathlib import Path
from features.yolov11 import YOLOv11Detector
from features.cv_detection import CVDetector, Circle

class DetectionService:
    def __init__(self, model_path="features/yolov11/weights/best.pt", input_size=640):
        """
        Initialize the detection service with YOLOv11 and CV detection
        """
        self.input_size = input_size
        
        # Initialize YOLOv11 detector
        self.yolo_detector = YOLOv11Detector(model_path, input_size)
        
        # Initialize CV detector
        self.cv_detector = CVDetector()
    
    def crop_region(self, image, bbox, padding=20):
        """
        Crop a region from the image with padding
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return image[y1:y2, x1:x2]
    
    def detect_coins_in_crop(self, cropped_image):
        """
        Apply CV detection algorithm to cropped image to find precise coin boundaries
        """
        return self.cv_detector.detect(cropped_image)
    
    def transform_circles_to_original(self, circles, crop_offset):
        """
        Transform circle coordinates from cropped image to original image coordinates
        """
        x_offset, y_offset = crop_offset
        transformed_circles = []
        
        for circle in circles:
            transformed_circle = Circle(
                cx=circle.cx + x_offset,
                cy=circle.cy + y_offset,
                r=circle.r
            )
            transformed_circles.append(transformed_circle)
        
        return transformed_circles
    
    def detect_image(self, image_path):
        """
        Process a single image with hybrid detection
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detected circles
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Step 1: Detect objects with YOLOv11
        print("Step 1: Running YOLOv11 detection...")
        yolo_detections = self.yolo_detector.detect(image)
        print(f"YOLOv11 detected {len(yolo_detections)} potential coin regions")
        
        # Step 2: Process each detected region with CV detection algorithm
        print("Step 2: Processing each region with CV detection algorithm...")
        all_circles = []
        
        for i, (x1, y1, x2, y2, conf) in enumerate(yolo_detections):
            print(f"  Processing region {i+1}/{len(yolo_detections)} (confidence: {conf:.3f})")
            
            # Crop the region
            cropped = self.crop_region(image, (x1, y1, x2, y2))
            
            if cropped.size == 0:
                print(f"    Warning: Empty crop for region {i+1}")
                continue
            
            # Apply detect.py algorithm to cropped region
            crop_mask, circles = self.detect_coins_in_crop(cropped)
            
            if circles:
                # Keep only the largest circle (by radius) from this region
                largest_circle = max(circles, key=lambda c: c.r)
                
                # Transform the largest circle to original image coordinates
                transformed_circle = self.transform_circles_to_original([largest_circle], (x1, y1))[0]
                all_circles.append(transformed_circle)
                print(f"    Found {len(circles)} circles, kept largest (radius: {largest_circle.r:.1f})")
            else:
                print(f"    No coins found in this region")
        
        print(f"Total coins detected: {len(all_circles)}")
        
        return all_circles
    
    def create_mask(self, image_shape, circles):
        """
        Create binary mask from detected circles
        
        Args:
            image_shape: (height, width) of original image
            circles: List of detected circles
            
        Returns:
            Binary mask of detected coins
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for circle in circles:
            cv2.circle(mask, (int(circle.cx), int(circle.cy)), int(circle.r), 255, thickness=-1)
        
        return mask
