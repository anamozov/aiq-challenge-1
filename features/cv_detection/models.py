"""
Data classes for CV-based detection.

"""

from dataclasses import dataclass


@dataclass
class Circle:
    """
    Represents a detected circle with center coordinates and radius.
    
    Attributes:
        cx: X coordinate of circle center
        cy: Y coordinate of circle center  
        r: Radius of the circle
    """
    cx: float
    cy: float
    r: float
    @property
    def bbox(self):
        """
        Get bounding box coordinates for the circle.
        
        Returns:
            Tuple of (x, y, width, height) representing the bounding box
        """
        x = int(max(self.cx - self.r, 0))
        y = int(max(self.cy - self.r, 0))
        w = int(self.r * 2)
        h = int(self.r * 2)
        return x, y, w, h
    
    @property
    def center(self):
        """
        Get center coordinates as a tuple.
        
        Returns:
            Tuple of (cx, cy) representing the center
        """
        return (self.cx, self.cy)
    
    @property
    def area(self):
        """
        Calculate the area of the circle.
        
        Returns:
            Area of the circle (π * r²)
        """
        import math
        return math.pi * self.r * self.r
