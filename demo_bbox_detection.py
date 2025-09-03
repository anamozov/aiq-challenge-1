#!/usr/bin/env python3
"""
Demo script showing the updated hybrid detection with bounding boxes.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_detection import HybridDetector


def demo_bbox_detection():
    """Demo the hybrid detection system with bounding boxes."""
    
    # Test image path
    test_image = "/home/anamozov/data/codes/mine/AIQ/coin-dataset/8eb6dc58-bab0-4378-8209-b17c4d9c74a1_jpg.rf.50d3b31c53105ed1dd7a5f70fa56603d.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return False
    
    try:
        # Initialize detector
        print("Initializing hybrid detector...")
        detector = HybridDetector()
        
        # Process the image
        print("Processing image...")
        original_image, final_mask, all_circles = detector.process_image(test_image)
        
        # Print results
        print(f"\nDetection Results:")
        print(f"Total coins detected: {len(all_circles)}")
        
        if all_circles:
            print("\nCircle details with bounding boxes:")
            for i, circle in enumerate(all_circles):
                x, y, w, h = circle.bbox
                print(f"  Circle {i+1}: center=({circle.cx:.1f}, {circle.cy:.1f}), radius={circle.r:.1f}")
                print(f"    Bounding box: x={x}, y={y}, width={w}, height={h}")
        
        print("\nDemo completed successfully!")
        print("The combined image now shows:")
        print("- Left side: Original image with green bounding boxes around detected coins")
        print("- Right side: Binary mask of detected coins")
        print("- Each coin is numbered and has a red center point")
        
        return True
        
    except Exception as e:
        print(f"Error during demo: {e}")
        return False


if __name__ == "__main__":
    success = demo_bbox_detection()
    sys.exit(0 if success else 1)
