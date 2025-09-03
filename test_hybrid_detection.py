#!/usr/bin/env python3
"""
Test script for hybrid detection.
This script demonstrates how to use the hybrid detection system.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_detection import HybridDetector


def test_hybrid_detection():
    """Test the hybrid detection system."""
    
    # Check if model exists
    model_path = "app/yolov11/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Please train the YOLOv11 model first or provide the correct path.")
        return False
    
    # Test image path (you can change this to any image you want to test)
    test_image = "/home/anamozov/data/codes/mine/AIQ/coin-dataset/af9912fe-a67a-4f48-9a9c-221a569cb210_jpg.rf.e3697c3ab17d9720727542f7bae8753c.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        print("Please provide a valid image path.")
        return False
    
    try:
        # Initialize detector
        print("Initializing hybrid detector...")
        detector = HybridDetector(model_path)
        
        # Process the image
        print("Processing image...")
        original_image, final_mask, all_circles = detector.process_image(test_image)
        
        # Print results
        print(f"\nDetection Results:")
        print(f"Total coins detected: {len(all_circles)}")
        
        if all_circles:
            print("\nCircle details:")
            for i, circle in enumerate(all_circles):
                print(f"  Circle {i+1}: center=({circle.cx:.1f}, {circle.cy:.1f}), radius={circle.r:.1f}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False


if __name__ == "__main__":
    success = test_hybrid_detection()
    sys.exit(0 if success else 1)
