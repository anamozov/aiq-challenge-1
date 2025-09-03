#!/usr/bin/env python3
"""
Visual evaluation script that processes all images from a dataset directory,
performs coin detection, draws bounding boxes, creates masks, and saves
combined images with original and mask side by side.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os
import argparse
from typing import List, Tuple

# Add the app directory to the path so we can import the detection pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from pipeline.detect import segment_circles_and_mask, Circle, create_combined_image


def draw_bounding_boxes(image_bgr: np.ndarray, circles: List[Circle]) -> np.ndarray:
    """
    Draw bounding boxes and center points on the image.
    
    Args:
        image_bgr: Input BGR image
        circles: List of detected circles
        
    Returns:
        Image with bounding boxes and center points drawn
    """
    display_img = image_bgr.copy()
    
    if circles:
        for i, circle in enumerate(circles):
            # Draw bounding box
            x, y, w, h = circle.bbox
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw circle outline
            cv2.circle(display_img, (int(circle.cx), int(circle.cy)), int(circle.r), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(display_img, (int(circle.cx), int(circle.cy)), 3, (0, 0, 255), -1)
            
            # Add circle number label
            cv2.putText(display_img, f"{i+1}", 
                       (int(circle.cx) - 10, int(circle.cy) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return display_img


def create_combined_image(image_bgr: np.ndarray, mask: np.ndarray, circles: List[Circle] = None) -> np.ndarray:
    """
    Create a combined image with original (with bboxes and circles) and mask side by side.
    
    Args:
        image_bgr: Original BGR image
        mask: Binary mask of detected circles
        circles: List of detected circles
        
    Returns:
        Combined image with original and mask side by side
    """
    # Create image with bounding boxes and circles
    bbox_img = draw_bounding_boxes(image_bgr, circles)
    
    # Convert mask to 3-channel for concatenation
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Resize images to the same height for side-by-side display
    height = max(bbox_img.shape[0], mask_colored.shape[0])
    
    # Resize both images to the same height
    bbox_img_resized = cv2.resize(bbox_img, (int(bbox_img.shape[1] * height / bbox_img.shape[0]), height))
    mask_resized = cv2.resize(mask_colored, (int(mask_colored.shape[1] * height / mask_colored.shape[0]), height))
    
    # Concatenate images horizontally: original with bboxes and circles | mask
    combined = np.hstack((bbox_img_resized, mask_resized))
    
    # Add section labels
    cv2.putText(combined, "Original + Detection", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Detection Mask", (bbox_img_resized.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return combined


def process_single_image(image_path: Path, output_dir: Path, suffix: str = "_detection") -> bool:
    """
    Process a single image: detect circles, create mask, and save combined result.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output images
        suffix: Suffix to add to output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Warning: Could not load {image_path}")
            return False
        
        print(f"Processing: {image_path.name}")
        
        # Detect circles and create mask
        mask, circles = segment_circles_and_mask(image_bgr)
        
        # Create combined image
        combined = create_combined_image(image_bgr, mask, circles)
        
        # Generate output filename
        stem = image_path.stem
        extension = image_path.suffix
        output_filename = f"{stem}{suffix}{extension}"
        output_path = output_dir / output_filename
        
        # Save combined image
        cv2.imwrite(str(output_path), combined)
        
        print(f"  Detected {len(circles) if circles else 0} circles")
        print(f"  Saved: {output_path.name}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def process_dataset(input_dir: str, output_dir: str = None, suffix: str = "_detection", 
                   image_extensions: List[str] = None) -> None:
    """
    Process all images in a dataset directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images (default: input_dir + suffix)
        suffix: Suffix to add to output filenames
        image_extensions: List of image file extensions to process
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Set output directory
    if output_dir is None:
        output_path = input_path.parent / f"{input_path.name}{suffix}"
    else:
        output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_path.name}")
        
        if process_single_image(image_path, output_path, suffix):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visual evaluation script for coin detection")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input directory containing images")
    parser.add_argument("--output", "-o", 
                       help="Output directory (default: input_dir + suffix)")
    parser.add_argument("--suffix", "-s", default="_detection",
                       help="Suffix to add to output filenames (default: _detection)")
    parser.add_argument("--extensions", "-e", nargs="+", 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
                       help="Image file extensions to process")
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        suffix=args.suffix,
        image_extensions=args.extensions
    )


if __name__ == "__main__":
    main()
