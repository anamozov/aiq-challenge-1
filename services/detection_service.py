"""
Detection service for AIQ Challenge 1 -  detection logic migrated from hybrid_detection.py
"""

import cv2
import numpy as np
from pathlib import Path
from features.yolov11 import YOLOv11Detector
from features.cv_detection import CVDetector, Circle
from core.logging_config import get_logger

# Set up logger
logger = get_logger("detection")

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
                r=circle.r,
            )
            transformed_circles.append(transformed_circle)
        
        return transformed_circles
    # return confidence for each bbox which has a circle
    def detect_image(self, image_path):
        """
        Process a single image with hybrid detection
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (yolo_detections, circles_with_confidence)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        logger.info(f"Processing image: {image_path}")
        logger.debug(f"Image shape: {image.shape}")
        
        # Step 1: Detect objects with YOLOv11
        logger.info("Running YOLOv11 detection...")
        yolo_detections = self.yolo_detector.detect(image)
        logger.info(f"YOLOv11 detected {len(yolo_detections)} potential coin regions")
        
        # Step 2: Process each detected region with CV detection algorithm
        logger.info("Processing each region with CV detection algorithm...")
        all_circles_with_confidence = []
        
        for i, (x1, y1, x2, y2, conf) in enumerate(yolo_detections):
            logger.debug(f"Processing region {i+1}/{len(yolo_detections)} (confidence: {conf:.3f})")
            
            # Crop the region (this modifies x1, y1, x2, y2 with padding)
            cropped = self.crop_region(image, (x1, y1, x2, y2))
            
            if cropped.size == 0:
                logger.warning(f"Empty crop for region {i+1}")
                continue
            
            # Apply detect.py algorithm to cropped region
            crop_mask, circles = self.detect_coins_in_crop(cropped)
            
            if circles:
                # Keep only the largest circle (by radius) from this region
                largest_circle = max(circles, key=lambda c: c.r)
                
                # Transform the largest circle to original image coordinates
                # Use the actual crop coordinates (after padding) as offset
                transformed_circle = self.transform_circles_to_original([largest_circle], (x1, y1))[0]
                all_circles_with_confidence.append((transformed_circle, conf))
                logger.debug(f"Found {len(circles)} circles, kept largest (radius: {largest_circle.r:.1f})")
            else:
                logger.debug(f"No coins found in this region")
        
        logger.info(f"Total coins detected: {len(all_circles_with_confidence)}")
        
        return yolo_detections, all_circles_with_confidence
    
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
    
    def draw_bounding_boxes(self, image, circles):
        """
        Draw bounding boxes and center points on the image
        
        Args:
            image: Input BGR image
            circles: List of detected circles
            
        Returns:
            Image with bounding boxes and center points drawn
        """
        display_img = image.copy()
        
        if circles:
            for i, circle in enumerate(circles):
                # Ensure coordinates are within image bounds
                cx = int(circle.cx)
                cy = int(circle.cy)
                r = int(circle.r)
                
                # Draw bounding box (ensure it's within image bounds)
                x1 = max(0, cx - r)
                y1 = max(0, cy - r)
                x2 = min(image.shape[1], cx + r)
                y2 = min(image.shape[0], cy + r)
                
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw circle outline
                cv2.circle(display_img, (cx, cy), r, (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(display_img, (cx, cy), 3, (0, 0, 255), -1)
                
                # Add circle number label
                label_x = max(10, cx - 10)
                label_y = max(20, cy - 10)
                cv2.putText(display_img, f"{i+1}", 
                           (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_img
    
    def draw_yolo_boxes(self, image, yolo_detections):
        """
        Draw YOLO bounding boxes on the image
        
        Args:
            image: Input BGR image
            yolo_detections: List of YOLO detections (x1, y1, x2, y2, confidence)
            
        Returns:
            Image with YOLO bounding boxes drawn
        """
        display_img = image.copy()
        
        if yolo_detections:
            for i, (x1, y1, x2, y2, conf) in enumerate(yolo_detections):
                # Draw bounding box rectangle
                cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add confidence and number label
                label = f"{i+1}: {conf:.2f}"
                cv2.putText(display_img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return display_img
    
    def create_combined_image(self, image, mask, yolo_detections=None, circles=None):
        """
        Create combined image showing original with YOLO bounding boxes on left, mask on right
        
        Args:
            image: Original BGR image
            mask: Binary mask of detected objects
            yolo_detections: List of YOLO detections (x1, y1, x2, y2, confidence)
            circles: List of detected circles (optional)
            
        Returns:
            Combined image with YOLO boxes on left, mask on right
        """
        # Create image with YOLO bounding boxes drawn (no circles)
        if yolo_detections:
            detection_img = self.draw_yolo_boxes(image, yolo_detections)
        else:
            detection_img = image.copy()
        
        # Convert mask to 3-channel for display
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Resize images to the same height for side-by-side display
        height = max(detection_img.shape[0], mask_colored.shape[0])
        
        # Resize both images to the same height
        detection_img_resized = cv2.resize(detection_img, (int(detection_img.shape[1] * height / detection_img.shape[0]), height))
        mask_resized = cv2.resize(mask_colored, (int(mask_colored.shape[1] * height / mask_colored.shape[0]), height))
        
        # Concatenate images horizontally
        combined = np.hstack((detection_img_resized, mask_resized))
        
        # Add labels
        cv2.putText(combined, "YOLO Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Circle Mask", (detection_img_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined
    
    def save_detection_results(self, image_path, yolo_detections, circles, output_dir):
        """
        Save detection results including original image with YOLO bounding boxes, mask, and combined image
        
        Args:
            image_path: Path to original image
            yolo_detections: List of YOLO detections (x1, y1, x2, y2, confidence)
            circles: List of detected circles
            output_dir: Directory to save results
            
        Returns:
            Dictionary with paths to saved files
        """
        import os
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Generate output filenames
        image_name = Path(image_path).stem
        image_id = Path(image_path).name.split('_')[0] if '_' in Path(image_path).name else image_name
        
        # Create mask from circles
        mask = self.create_mask(image.shape, circles)
        
        # Create images
        detection_img = self.draw_yolo_boxes(image, yolo_detections)
        combined_img = self.create_combined_image(image, mask, yolo_detections, circles)
        
        # Save files
        result_files = {}
        
        # Original with YOLO bounding boxes
        detection_path = os.path.join(output_dir, f"{image_id}_detection.jpg")
        cv2.imwrite(detection_path, detection_img)
        result_files['detection'] = detection_path
        
        # Mask
        mask_path = os.path.join(output_dir, f"{image_id}_mask.jpg")
        cv2.imwrite(mask_path, mask)
        result_files['mask'] = mask_path
        
        # Combined image
        combined_path = os.path.join(output_dir, f"{image_id}_combined.jpg")
        cv2.imwrite(combined_path, combined_img)
        result_files['combined'] = combined_path
        
        return result_files