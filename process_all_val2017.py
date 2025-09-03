#!/usr/bin/env python3
"""
Process all images from val2017.txt using hybrid detection.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_detection import HybridDetector


def process_all_val2017():
    """Process all images from val2017.txt file."""
    
    # Paths
    val2017_txt = "app/yolov11/Dataset/COCO/val2017.txt"
    dataset_base = "app/yolov11/Dataset/COCO"
    output_dir = "improved_result"
    
    # Check if val2017.txt exists
    if not os.path.exists(val2017_txt):
        print(f"Error: {val2017_txt} not found")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Read image list
    with open(val2017_txt, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(image_paths)} images to process")
    
    # Initialize detector
    try:
        print("Initializing hybrid detector...")
        detector = HybridDetector()
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return False
    
    # Process each image
    successful = 0
    failed = 0
    
    for i, relative_path in enumerate(image_paths):
        # Construct full path
        full_path = os.path.join(dataset_base, relative_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: Image not found: {full_path}")
            failed += 1
            continue
        
        print(f"\nProcessing {i+1}/{len(image_paths)}: {os.path.basename(full_path)}")
        
        try:
            # Process the image
            original_image, final_mask, all_circles, yolo_detections = detector.process_image(full_path)
            
            # Create combined visualization
            combined_image = detector.create_combined_image_with_yolo_boxes(original_image, final_mask, yolo_detections)
            
            # Save result
            image_name = Path(full_path).stem
            output_file = Path(output_dir) / f"{image_name}_result.png"
            import cv2
            cv2.imwrite(str(output_file), combined_image)
            
            print(f"  ✓ Saved: {output_file}")
            print(f"  ✓ YOLO detected: {len(yolo_detections)} regions")
            print(f"  ✓ Final coins: {len(all_circles)}")
            
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {full_path}: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Processing completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(image_paths)}")
    print(f"Results saved in: {output_dir}/")
    
    return successful > 0


if __name__ == "__main__":
    success = process_all_val2017()
    sys.exit(0 if success else 1)
