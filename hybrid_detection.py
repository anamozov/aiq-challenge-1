#!/usr/bin/env python3
"""
Hybrid Detection Script
Combines YOLOv11 model detection with detect.py algorithm for precise coin extraction.

This script:
1. Uses YOLOv11 to detect potential coin regions
2. Crops each detected region
3. Applies detect.py algorithm to each cropped region for precise coin extraction
4. Creates masks in the original image coordinates
5. Displays results with original image and mask side by side

Usage:
    python3 hybrid_detection.py <image_path> [--model-path weights/best.pt] [--output-dir results/]
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.yolov11 import YOLOv11Detector
from features.cv_detection import CVDetector, Circle



class HybridDetector:
    def __init__(self, model_path="features/yolov11/weights/best.pt", input_size=640):
        """
        Initialize the hybrid detector with YOLOv11 and CV detection.
        
        Args:
            model_path: Path to the trained YOLOv11 model
            input_size: Input size for YOLOv11 model
        """
        self.input_size = input_size
        
        # Initialize YOLOv11 detector
        self.yolo_detector = YOLOv11Detector(model_path, input_size)
        
        # Initialize CV detector
        self.cv_detector = CVDetector()
        
    
    def crop_region(self, image, bbox, padding=20):
        """
        Crop a region from the image with padding.
        
        Args:
            image: Input image
            bbox: (x1, y1, x2, y2) bounding box
            padding: Padding around the bounding box
            
        Returns:
            Cropped image
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
        Apply CV detection algorithm to cropped image to find precise coin boundaries.
        
        Returns:
            mask: Binary mask of detected coins
            circles: List of detected circles
        """
        return self.cv_detector.detect(cropped_image)
    
    def transform_circles_to_original(self, circles, crop_offset):
        """
        Transform circle coordinates from cropped image to original image coordinates.
        
        Args:
            circles: List of circles in cropped image coordinates
            crop_offset: (x_offset, y_offset) of the crop in original image
            
        Returns:
            List of circles in original image coordinates
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
    
    def create_final_mask(self, image_shape, all_circles):
        """
        Create final mask in original image coordinates.
        
        Args:
            image_shape: (height, width) of original image
            all_circles: List of all detected circles in original coordinates
            
        Returns:
            Binary mask of all detected coins
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for circle in all_circles:
            cv2.circle(mask, (int(circle.cx), int(circle.cy)), int(circle.r), 255, thickness=-1)
        
        return mask
    
    def create_combined_image_with_yolo_boxes(self, image, mask, yolo_detections):
        """
        Create combined image with YOLO bounding boxes on the left and mask on the right.
        
        Args:
            image: Original image
            mask: Binary mask of detected coins
            yolo_detections: List of YOLO detections (x1, y1, x2, y2, confidence)
            
        Returns:
            Combined image
        """
        # Create a copy of the original image to draw YOLO bounding boxes on
        display_img = image.copy()
        
        # Draw YOLO bounding boxes
        for i, (x1, y1, x2, y2, conf) in enumerate(yolo_detections):
            # Draw bounding box rectangle
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add confidence and number label
            label = f"{i+1}: {conf:.2f}"
            cv2.putText(display_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Convert mask to 3-channel for concatenation
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Resize images to the same height for side-by-side display
        height = max(display_img.shape[0], mask_colored.shape[0])
        
        # Resize both images to the same height
        display_img_resized = cv2.resize(display_img, (int(display_img.shape[1] * height / display_img.shape[0]), height))
        mask_resized = cv2.resize(mask_colored, (int(mask_colored.shape[1] * height / mask_colored.shape[0]), height))
        
        # Concatenate images horizontally
        combined = np.hstack((display_img_resized, mask_resized))
        
        # Add labels
        cv2.putText(combined, "YOLO Detection + Precise Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Detected Circles Mask", (display_img_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined
    
    def process_image(self, image_path):
        """
        Process a single image with hybrid detection.
        
        Args:
            image_path: Path to input image
            
        Returns:
            original_image: Original input image
            final_mask: Binary mask of detected coins
            all_circles: List of all detected circles
            yolo_detections: List of YOLO detections (x1, y1, x2, y2, confidence)
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
        
        # Step 3: Create final mask
        print("Step 3: Creating final mask...")
        final_mask = self.create_final_mask(image.shape, all_circles)
        
        print(f"Total coins detected: {len(all_circles)}")
        
        return image, final_mask, all_circles, yolo_detections


def main():
    parser = argparse.ArgumentParser(description='Hybrid coin detection using YOLOv11 + detect.py')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--model-path', default='features/yolov11/weights/best.pt', help='Path to YOLOv11 model')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--input-size', type=int, default=640, help='YOLOv11 input size')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize detector
    try:
        detector = HybridDetector(args.model_path, args.input_size)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the YOLOv11 model is trained and available at the specified path.")
        return 1
    
    # Process image
    try:
        original_image, final_mask, all_circles, yolo_detections = detector.process_image(args.image_path)
        
        # Create combined visualization using YOLO bounding boxes
        combined_image = detector.create_combined_image_with_yolo_boxes(original_image, final_mask, yolo_detections)
        
        # Save only the combined result
        image_name = Path(args.image_path).stem
        output_file = output_dir / f"{image_name}_result.png"
        cv2.imwrite(str(output_file), combined_image)
        
        # Print results
        print(f"\nResult saved to: {output_file}")
        print(f"Combined image: {image_name}_result.png")
        
        # Print circle details
        if all_circles:
            print(f"\nDetected circles:")
            for i, circle in enumerate(all_circles):
                print(f"  Circle {i+1}: center=({circle.cx:.1f}, {circle.cy:.1f}), radius={circle.r:.1f}")
        else:
            print("\nNo circles detected.")
        
        return 0
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
