from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Circle:
    cx: float
    cy: float
    raduis: float

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
    Strategy: color threshold to remove brown background, then HoughCircles.
    """
    img = image_bgr
    h, w = img.shape[:2]

    # Convert to HSV to separate brown background; tune thresholds generously
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Brown range (approx). Adjust as needed for dataset variability.
    lower_brown = np.array([5, 50, 20], dtype=np.uint8)
    upper_brown = np.array([25, 255, 255], dtype=np.uint8)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Foreground is inverse of brown
    fg_mask = cv2.bitwise_not(brown_mask)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Use masked image for circle detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    # Focus detection on foreground by zeroing background
    gray_fg = cv2.bitwise_and(gray, gray, mask=fg_mask)

    # HoughCircles parameters tuned for medium-sized circles
    circles = cv2.HoughCircles(
        gray_fg,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, min(h, w) // 20),
        param1=100,
        param2=20,
        minRadius=5,
        maxRadius=0,
    )

    result: list[Circle] = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for x, y, r in circles:
            result.append(Circle(cx=float(x), cy=float(y), r=float(r)))

    # Optionally, refine mask by drawing detected circles solid
    mask = np.zeros((h, w), dtype=np.uint8)
    for c in result:
        cv2.circle(mask, (int(c.cx), int(c.cy)), int(c.r), 255, thickness=-1)

    # If no circles, fallback to foreground mask
    if not result:
        mask = fg_mask

    return mask, result


