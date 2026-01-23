"""
Page boundary detection module.

Detects the 4-corner polygon boundary of a single page,
ensuring all content is strictly inside the polygon.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class PageBoundary:
    """Detected page boundary as a 4-corner polygon."""
    corners: np.ndarray  # Shape (4, 2): TL, TR, BR, BL
    confidence: float
    content_rect: Tuple[int, int, int, int]  # Bounding box of content (x, y, w, h)


class PageDetector:
    """
    Detects the boundary of a page as a 4-corner polygon using robust blob detection.
    
    Strategy:
    1. Downscale & Blur for noise reduction
    2. Threshold (Otsu) to separate light paper from dark background
    3. Find largest contour (the page blob)
    4. Fit tight polygon/hull to the blob
    """
    
    def __init__(self, margin_percent: float = 0.0):
        """
        Args:
            margin_percent: Safety margin as percentage of page size. 
                          Defaults to 0.0 for tightest crop.
        """
        self.margin_percent = margin_percent
        self._debug_info = {}
    
    def detect(self, image: np.ndarray, page_num: int = 1) -> PageBoundary:
        """
        Detect the page boundary polygon.
        
        Args:
            image: RGB image of a single page
            page_num: 1 for left page, 2 for right page (used for side-aware clamping)
            
        Returns:
            PageBoundary with 4-corner polygon
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # 1. Preprocessing (Downscale for speed + robustness)
        scale = 0.2
        small_h, small_w = int(h * scale), int(w * scale)
        if small_h == 0 or small_w == 0:
             scale = 1.0
             small = gray
        else:
            small = cv2.resize(gray, (small_w, small_h))
            
        # Blur to remove text details/noise
        blur = cv2.GaussianBlur(small, (15, 15), 0)
        
        # 2. Thresholding
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Morphological Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 4. Contour Analysis
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            return PageBoundary(corners, 0.0, (0, 0, w, h))
            
        largest_cnt = max(contours, key=cv2.contourArea)
        
        # 5. Fit Polygon
        largest_cnt = largest_cnt.astype(np.float32) * (1.0/scale)
        largest_cnt = largest_cnt.astype(np.int32)
        
        hull = cv2.convexHull(largest_cnt)
        pts = hull.squeeze()
        if len(pts.shape) == 1: 
             pts = pts.reshape(1, 2)
             
        # Extract extreme corners
        tl_idx = np.argmin(pts.sum(axis=1))
        br_idx = np.argmax(pts.sum(axis=1))
        diff = np.diff(pts, axis=1)
        tr_idx = np.argmin(diff)
        bl_idx = np.argmax(diff)
        
        corners = np.array([
            pts[tl_idx],
            pts[tr_idx], 
            pts[br_idx],
            pts[bl_idx]
        ], dtype=np.float32)
            
        corners = self._order_corners(corners)
        
        # 6. Text-Aware Refinement (Side-Aware Clamping)
        text_rect = self._detect_text_bbox(image, largest_cnt, page_num)
        if text_rect:
            corners = self._clamp_to_text(image, corners, text_rect, page_num)
        
        corners = self._clip_to_bounds(corners, w, h)
        corners = self._expand_corners(corners, w, h)
        
        x, y, cw, ch = cv2.boundingRect(largest_cnt)
        content_rect = (x, y, cw, ch)
        
        return PageBoundary(
            corners=corners,
            confidence=0.9,
            content_rect=content_rect
        )

    def _detect_text_bbox(self, image: np.ndarray, blob_cnt: np.ndarray, page_num: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the bounding box of text content inside the page blob.
        Uses edge density and side-aware masking to exclude page stacks.
        """
        h, w = image.shape[:2]
        scale = 0.5
        small_h, small_w = int(h * scale), int(w * scale)
        if small_h == 0 or small_w == 0:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # 1. Create Mask from Blob
        cnt_small = (blob_cnt.astype(np.float32) * scale).astype(np.int32)
        mask = np.zeros((small_h, small_w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt_small], -1, 255, -1)
        
        # 2. Text Detection (Adaptive Threshold)
        gray_small = cv2.resize(gray, (small_w, small_h))
        # Use a slightly more aggressive threshold to capture sparse text
        text_bin = cv2.adaptiveThreshold(gray_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 8)
        
        masked_text = cv2.bitwise_and(text_bin, text_bin, mask=mask)
        
        # 3. Morphological Cleanup (Kill thin vertical stack lines)
        kernel_h_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        text_no_stack = cv2.morphologyEx(masked_text, cv2.MORPH_OPEN, kernel_h_open)
        
        conn_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        connected = cv2.dilate(text_no_stack, conn_kernel, iterations=2)
        
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
        cleaned = cv2.morphologyEx(connected, cv2.MORPH_OPEN, clean_kernel)
        
        # 4. Vertical Edge Analysis (for masking stack)
        edges_v = cv2.Sobel(gray_small, cv2.CV_64F, 1, 0, ksize=3)
        edges_v_abs = np.abs(edges_v).astype(np.uint8)
        edges_masked = cv2.bitwise_and(edges_v_abs, edges_v_abs, mask=mask)
        edge_cols = np.sum(edges_masked, axis=0).astype(np.float32)
        edge_cols_smooth = cv2.GaussianBlur(edge_cols.reshape(1, -1), (31, 1), 0)[0]
        
        # Side-aware stack removal
        is_page_1 = (page_num == 1)
        mean_density = np.mean(edge_cols_smooth)
        mask_zx_min, mask_zx_max = 0, small_w
        
        if is_page_1: # Outer is Left
            outer_window = edge_cols_smooth[:small_w // 2]
            if len(outer_window) > 20:
                # 1. Find the primary peak (stack) in the first 20%
                peak_idx = np.argmax(outer_window[:small_w // 5])
                # 2. Find the deepest valley (margin) after the stack
                valley_idx = peak_idx + np.argmin(edge_cols_smooth[peak_idx : small_w // 2])
                # 3. Apply mask with safety buffer and a hard minimum 5% margin
                mask_zx_min = max(int(small_w * 0.05), valley_idx + 5)
        else: # Outer is Right
            if small_w > 40:
                peak_idx_rel = np.argmax(edge_cols_smooth[int(small_w * 4/5):])
                peak_idx = int(small_w * 4/5) + peak_idx_rel
                valley_idx = int(small_w // 2) + np.argmin(edge_cols_smooth[small_w // 2 : peak_idx])
                mask_zx_max = min(int(small_w * 0.95), valley_idx - 5)

        # Apply column mask to the cleaned text
        col_mask = np.zeros_like(cleaned)
        col_mask[:, mask_zx_min:mask_zx_max] = 255
        cleaned = cv2.bitwise_and(cleaned, col_mask)
        
        self._debug_info['mask_zx_min'] = mask_zx_min
        self._debug_info['mask_zx_max'] = mask_zx_max

        # 5. Find Content Bounds
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        all_pts = []
        for c in contours:
            rect = cv2.boundingRect(c)
            # Filter noise - require minimum width and reasonable height
            if rect[2] > 20 and rect[3] > 10 and rect[3] < small_h * 0.9:
                all_pts.append(c)
                
        if not all_pts:
            return None
            
        all_pts = np.vstack([c.reshape(-1, 2) for c in all_pts])
        x, y, cw, ch = cv2.boundingRect(all_pts)
        
        # Scale back up
        inv_scale = 1.0 / scale
        return int(x * inv_scale), int(y * inv_scale), int(cw * inv_scale), int(ch * inv_scale)

    def _clamp_to_text(self, image: np.ndarray, corners: np.ndarray, text_rect: Tuple[int, int, int, int], page_num: int) -> np.ndarray:
        """
        Aggressively clamp polygon corners to exclude page stacks using contour-based edge detection.
        Side-aware: prioritizes outer edge for stack removal while keeping spine edge.
        """
        h, w = image.shape[:2]
        tx, ty, tw, th = text_rect
        text_x_min, text_x_max = tx, tx + tw
        blob_x_min = np.min(corners[:, 0])
        blob_x_max = np.max(corners[:, 0])
        
        left_margin = text_x_min - blob_x_min
        right_margin = blob_x_max - text_x_max
        
        new_min_x = blob_x_min
        new_max_x = blob_x_max

        is_page_1 = (page_num == 1)
        
        # Convert to grayscale for edge detection if needed
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Helper to find innermost vertical edge in a margin
        def find_innermost_edge(roi_x_min, roi_x_max, orientation='left'):
            if roi_x_max <= roi_x_min + 5:
                return None
            
            # Extract ROI
            margin_roi = gray[:, max(0, roi_x_min):min(w, roi_x_max)]
            if margin_roi.shape[1] < 5:
                return None
                
            # Edge detection
            blur_roi = cv2.GaussianBlur(margin_roi, (5, 5), 0)
            edges = cv2.Canny(blur_roi, 50, 150)
            
            # Find contours in the edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            v_edges = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if ch > h * 0.1: # At least 10% of image height
                    aspect = ch / max(1, cw)
                    if aspect > 2.0:
                        if orientation == 'left':
                            v_edges.append(roi_x_min + x + cw) 
                        else:
                            v_edges.append(roi_x_min + x)
            
            if not v_edges:
                return None
                
            # Return the innermost edge (closest to text)
            return max(v_edges) if orientation == 'left' else min(v_edges)

        # --- Outer Side Clamping (The primary issue) ---
        # Force a very tight margin (max 40px) to ensure no stack is included.
        # This is the "Force-Clamp" rule.
        force_limit = 40.0

        if is_page_1: # Page 1: Outer is Left
            if left_margin > 5:
                # 1. Try to find paper edge
                inner_edge = find_innermost_edge(int(blob_x_min), int(text_x_min), 'left')
                if inner_edge is not None:
                    new_min_x = max(new_min_x, inner_edge + 5)
                
                # 2. Hard Force-Clamp (absolute safety)
                if text_x_min - new_min_x > force_limit:
                    new_min_x = text_x_min - force_limit
        else: # Page 2: Outer is Right
            if right_margin > 5:
                inner_edge = find_innermost_edge(int(text_x_max), int(blob_x_max), 'right')
                if inner_edge is not None:
                    new_max_x = min(new_max_x, inner_edge - 5)
                
                if new_max_x - text_x_max > force_limit:
                    new_max_x = text_x_max + force_limit

        # --- Spine Side Clamping (Keep it tight but manageable) ---
        if is_page_1: # Page 1: Spine is Right
            if right_margin > 40:
                new_max_x = min(new_max_x, text_x_max + 40)
        else: # Page 2: Spine is Left
            if left_margin > 40:
                new_min_x = max(new_min_x, text_x_min - 40)

        # Apply Clamping to corners
        # Force verticality for a professional look on the clamped sides
        for i in range(4):
            if corners[i, 0] > new_max_x:
                corners[i, 0] = new_max_x
            if corners[i, 0] < new_min_x:
                corners[i, 0] = new_min_x
            
        return corners

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: TL, TR, BR, BL."""
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1).flatten()
        
        tl = corners[np.argmin(s)]
        br = corners[np.argmax(s)]
        tr = corners[np.argmin(diff)]
        bl = corners[np.argmax(diff)]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)
        
    def _clip_to_bounds(self, corners: np.ndarray, w: int, h: int) -> np.ndarray:
        """Clip corners to image boundaries."""
        corners[:, 0] = np.clip(corners[:, 0], 0, w)
        corners[:, 1] = np.clip(corners[:, 1], 0, h)
        return corners
        
    def _expand_corners(self, corners: np.ndarray, w: int, h: int) -> np.ndarray:
        """Slightly expand polygon to ensure text safety."""
        if self.margin_percent <= 0:
            return corners
            
        center = np.mean(corners, axis=0)
        
        # Shift corners away from center
        expanded = corners + (corners - center) * self.margin_percent
        
        return self._clip_to_bounds(expanded, w, h)
    
    def get_debug_info(self) -> dict:
        """Get debug information."""
        return self._debug_info
    
    def get_preview(self, image: np.ndarray, boundary: PageBoundary) -> np.ndarray:
        """Create preview with polygon overlay."""
        preview = image.copy()
        
        # Draw polygon
        pts = boundary.corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [pts], True, (0, 255, 0), 10) # Thicker line for better visibility
        
        # Draw corners
        for i, (x, y) in enumerate(boundary.corners):
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][i]
            cv2.circle(preview, (int(x), int(y)), 20, color, -1)
        
        return preview
