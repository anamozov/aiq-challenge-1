from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Circle:
    cx: float
    cy: float
    r: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x = int(max(self.cx - self.r, 0))
        y = int(max(self.cy - self.r, 0))
        w = int(self.r * 2)
        h = int(self.r * 2)
        return x, y, w, h


def segment_circles_and_mask(image_bgr: np.ndarray):
    """
    Returns (mask_uint8, circles)
    - mask: 0 background, 255 foreground circles
    - circles: list of detected circles (cx, cy, r)
    Strategy: Multiple approaches to detect circular objects robustly.
    """
    img = image_bgr
    h, w = img.shape[:2]

    # Convert to grayscale and apply preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to better separate foreground from background
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours to identify potential circular objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and circularity
    min_area = (min(h, w) // 20) ** 2  # Minimum area based on image size
    max_area = (min(h, w) // 2) ** 2   # Maximum area based on image size
    
    circular_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Calculate circularity: 4π*area/perimeter²
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:  # Threshold for circularity
                    circular_contours.append(contour)
    
    # Create mask from circular contours
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, circular_contours, -1, 255, thickness=-1)
    
    # Also try HoughCircles with more restrictive parameters
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(h, w) // 4,  # Minimum distance between circle centers
        param1=50,               # Upper threshold for edge detection
        param2=30,               # Accumulator threshold for center detection
        minRadius=min(h, w) // 20,  # Minimum radius
        maxRadius=min(h, w) // 3,   # Maximum radius
    )
    
    result: list[Circle] = []
    
    # Add circles from HoughCircles if they don't overlap significantly with existing ones
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for x, y, r in circles:
            # Check if this circle overlaps significantly with existing ones
            overlaps = False
            for existing_circle in result:
                distance = np.sqrt((x - existing_circle.cx)**2 + (y - existing_circle.cy)**2)
                if distance < (r + existing_circle.r) * 0.8:  # 80% overlap threshold
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(Circle(cx=float(x), cy=float(y), r=float(r)))
    
    # If we found circles from HoughCircles, use them to create the mask
    if result:
        mask = np.zeros((h, w), dtype=np.uint8)
        for c in result:
            cv2.circle(mask, (int(c.cx), int(c.cy)), int(c.r), 255, thickness=-1)
    # Otherwise, use the contour-based mask
    
    return mask, result


def create_combined_image(image_bgr: np.ndarray, mask: np.ndarray, circles: list[Circle] = None):
    """
    Create a combined image with original and mask side by side.
    Also draw detected circles on the original image.
    """
    # Create a copy of the original image to draw circles on
    display_img = image_bgr.copy()
    
    # Draw detected circles on the image
    if circles:
        for circle in circles:
            cv2.circle(display_img, (int(circle.cx), int(circle.cy)), int(circle.r), (0, 255, 0), 2)
            # Draw center point
            cv2.circle(display_img, (int(circle.cx), int(circle.cy)), 3, (0, 0, 255), -1)
    
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
    cv2.putText(combined, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Detected Circles Mask", (display_img_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return combined


if __name__ == "__main__":
    # Load the image
    # lets do this for all images in the coin-dataset folder
    
    image_path = "/home/anamozov/data/codes/mine/AIQ/coin-dataset/af9912fe-a67a-4f48-9a9c-221a569cb210_jpg.rf.e3697c3ab17d9720727542f7bae8753c.jpg"
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        exit(1)
    
    # Detect circles and create mask
    mask, circles = segment_circles_and_mask(image_bgr)
    
    # Display results
    print(f"Detected {len(circles) if circles else 0} circles")
    if circles:
        for i, circle in enumerate(circles):
            print(f"Circle {i+1}: center=({circle.cx:.1f}, {circle.cy:.1f}), radius={circle.r:.1f}")
    
    # Create combined image with original and mask side by side
    combined = create_combined_image(image_bgr, mask, circles)
    
    # Save the results
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("combined_result.png", combined)
    print("Results saved as 'mask.png' and 'combined_result.png'")

