"""
Spine detection module for book spread images.

Detects the spine/gutter of an open book and splits the image into
separate left and right page images.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SpineDetectionResult:
    """Result of spine detection."""
    spine_x_top: int  # X coordinate of spine at top of image
    spine_x_bottom: int  # X coordinate of spine at bottom of image
    confidence: float  # 0-1 confidence score
    left_page: Optional[np.ndarray] = None  # Left page image
    right_page: Optional[np.ndarray] = None  # Right page image
    is_spread: bool = False  # True if this appears to be a two-page spread

    @property
    def spine_x_avg(self) -> int:
        """Average X coordinate."""
        return (self.spine_x_top + self.spine_x_bottom) // 2


class SpineDetector:
    """
    Detects the spine/gutter in book spread images.
    
    Uses classic CV techniques:
    1. Two-Page Blob Separation (Gap finding)
    2. Hough Line Detection (Fallback)
    """
    
    def __init__(self, search_margin: float = 0.20):
        """
        Args:
            search_margin: How far from center to search (0.20 = 20% each side)
        """
        self.search_margin = search_margin
        self._debug_info = {}
    
    def detect_and_split(self, image: np.ndarray) -> SpineDetectionResult:
        """
        Detect spine and split image into left/right pages.
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Check if this is actually a spread
        aspect_ratio = w / h
        is_spread = aspect_ratio > 1.1
        
        if not is_spread:
            return SpineDetectionResult(
                spine_x_top=w,
                spine_x_bottom=w,
                confidence=0.0,
                left_page=image,
                right_page=np.zeros((h, 1, 3), dtype=np.uint8),
                is_spread=False
            )
        
        # Find spine
        res = self._find_spine(gray)
        
        # Split image (using average for now, but pipeline will do better)
        split_x = res.spine_x_avg
        res.left_page = image[:, :split_x].copy()
        res.right_page = image[:, split_x:].copy()
        res.is_spread = True
        
        self._debug_info = {
            'spine_x_top': res.spine_x_top,
            'spine_x_bottom': res.spine_x_bottom,
            'confidence': res.confidence
        }
        
        return res
    
    def _find_spine(self, gray: np.ndarray) -> SpineDetectionResult:
        """
        Find slanted spine line using blob gap or Hough lines.
        """
        h, w = gray.shape
        center = w // 2
        
        # 1. Primary Method: Blob Gap at multiple slices
        scale = 0.2
        small_h, small_w = int(h * scale), int(w * scale)
        small = cv2.resize(gray, (small_w, small_h))
        
        blur = cv2.GaussianBlur(small, (15, 15), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(binary, kernel, iterations=2)
        
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        if len(contours) >= 2:
            c1, c2 = contours[0], contours[1]
            if cv2.contourArea(c2) > (small_w * small_h) * 0.05:
                # Analyze gap at different Y positions
                gap_points = []
                num_slices = 10
                for i in range(num_slices):
                    y = int(small_h * (i + 0.5) / num_slices)
                    
                    # Find span of blobs at this Y
                    row = eroded[y, :]
                    indices = np.where(row > 0)[0]
                    if len(indices) < 2: continue
                    
                    # Group indices into blobs? No, simpler: 
                    # Find the gap near the center.
                    # We know c1 and c2 are the two pages.
                    # Mask them individually
                    mask1 = np.zeros_like(eroded)
                    cv2.drawContours(mask1, [c1], -1, 255, -1)
                    mask2 = np.zeros_like(eroded)
                    cv2.drawContours(mask2, [c2], -1, 255, -1)
                    
                    row1 = mask1[y, :]
                    row2 = mask2[y, :]
                    
                    idx1 = np.where(row1 > 0)[0]
                    idx2 = np.where(row2 > 0)[0]
                    
                    if len(idx1) > 0 and len(idx2) > 0:
                        # Find gap between them
                        if idx1[-1] < idx2[0]: # 1 is left, 2 is right
                            gap_mid = (idx1[-1] + idx2[0]) / 2
                        elif idx2[-1] < idx1[0]: # 2 is left, 1 is right
                            gap_mid = (idx2[-1] + idx1[0]) / 2
                        else:
                            continue # Overlap or weird
                        
                        gap_points.append((gap_mid, y))
                
                if len(gap_points) >= 3:
                    # Fit a line to gap points
                    pts = np.array(gap_points)
                    vx, vy, x, y = [v.item() for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)]
                    
                    # Line equation: x = x0 + t*vx, y = y0 + t*vy
                    # At y=0: t = -y0/vy. x = x0 - (y0/vy)*vx
                    # At y=h: t = (small_h-y0)/vy. x = x0 + ((small_h-y0)/vy)*vx
                    
                    t_top = -y / vy
                    x_top = x + t_top * vx
                    
                    t_bottom = (small_h - y) / vy
                    x_bottom = x + t_bottom * vx
                    
                    spine_x_top = int(x_top / scale)
                    spine_x_bottom = int(x_bottom / scale)
                    
                    # Sanity check centrality
                    avg_x = (spine_x_top + spine_x_bottom) // 2
                    if abs(avg_x - w//2) < w * 0.25:
                        return SpineDetectionResult(spine_x_top, spine_x_bottom, 0.95)

        # 2. Hough Lines Fallback
        edges = cv2.Canny(blur, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=small_h//4, maxLineGap=20)
        
        if lines is not None:
            valid_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < small_w * 0.1: # Vertical-ish
                    if abs((x1+x2)/2 - small_w//2) < small_w//5:
                        valid_lines.append(line[0])
            
            if valid_lines:
                pts = []
                for x1, y1, x2, y2 in valid_lines:
                    pts.append([x1, y1])
                    pts.append([x2, y2])
                pts = np.array(pts, dtype=np.float32)
                vx, vy, x, y = [v.item() for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)]
                
                t_top = -y / vy
                x_top = x + t_top * vx
                t_bottom = (small_h - y) / vy
                x_bottom = x + t_bottom * vx
                
                return SpineDetectionResult(int(x_top/scale), int(x_bottom/scale), 0.8)

        # 3. Last Fallback
        return SpineDetectionResult(w//2, w//2, 0.3)
    
    def get_debug_info(self) -> dict:
        """Get debug information from last detection."""
        return self._debug_info
    
    def get_preview(self, image: np.ndarray, result: SpineDetectionResult) -> np.ndarray:
        """Create preview image with spine marked (possibly slanted)."""
        preview = image.copy()
        h = preview.shape[0]
        
        # Draw spine line (possibly slanted)
        cv2.line(preview, (result.spine_x_top, 0), (result.spine_x_bottom, h), 
                (0, 255, 0), 3)
        
        return preview
