"""
Enhanced page dewarping module with 2D surface modeling.

Improvements over basic dewarper:
- 2D surface modeling (X + Y curvature)
- Multi-method text line detection
- Spine/gutter detection
- Adaptive mesh density
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from pagescan.core.perspective import PerspectiveCorrector, PerspectiveParams


@dataclass
class EnhancedDewarpParams:
    """Parameters for enhanced dewarping."""
    # Detection
    text_threshold: int = 127
    min_line_length: int = 50
    line_gap: int = 15
    
    # Curve fitting
    num_control_points_x: int = 40
    num_control_points_y: int = 30
    smoothing_factor: float = 0.2
    
    # Mesh
    mesh_rows: int = 100
    mesh_cols: int = 100
    
    # Spine detection
    spine_position: Optional[str] = None  # 'left', 'right', 'center', or None for auto
    spine_falloff: float = 0.25  # How quickly curvature decreases from spine (0-1)
    
    # Correction strength
    vertical_strength: float = 2.5  # Increased for stronger correction
    horizontal_strength: float = 1.0
    
    # Book spread handling
    is_book_spread: Optional[bool] = None  # None = auto-detect, True/False = force
    
    # Auto-crop
    auto_crop: bool = True
    crop_margin: int = 50  # Pixels to keep around detected content


class EnhancedDewarper:
    """
    Enhanced dewarper with 2D surface modeling for book pages.
    
    Key improvements:
    - Models curvature in both X and Y directions
    - Detects spine/gutter position for asymmetric correction
    - Uses adaptive mesh with higher density near spine
    - Combines multiple detection methods
    """
    
    def __init__(self, params: Optional[EnhancedDewarpParams] = None):
        self.params = params or EnhancedDewarpParams()
        self._last_surface = None
        self._spine_x = None
        self._debug_info = {}
        # Initialize perspective corrector for cropping
        self.cropper = PerspectiveCorrector()
    
    def dewarp(self, image: np.ndarray, 
               params: Optional[EnhancedDewarpParams] = None) -> np.ndarray:
        """
        Dewarp a curved book page using 2D surface modeling.
        
        Args:
            image: Input image as RGB numpy array
            params: Optional parameters override
            
        Returns:
            Dewarped image
        """
        params = params or self.params
        h, w = image.shape[:2]
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Auto-crop if enabled
        if params.auto_crop:
            cropped_image, crop_info = self.crop_to_content(image, params)
            if cropped_image is not None and crop_info is not None:
                image = cropped_image
                # Update gray image for cropped content
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image.copy()
                self._debug_info['crop_info'] = crop_info
        
        h, w = image.shape[:2]
        
        # Preprocess
        gray = self._preprocess(gray)
        
        # Detect if this is a book spread (two pages)
        is_spread = self._is_book_spread(gray, params)
        self._debug_info['is_book_spread'] = is_spread
        
        if is_spread:
            # Process each page separately then combine
            return self._dewarp_book_spread(image, gray, params)
        
        # Single page processing
        # Detect spine position
        spine_x = self._detect_spine(gray, params)
        self._spine_x = spine_x
        
        # Detect text lines using multiple methods
        text_lines = self._detect_text_lines_ensemble(gray, params)
        
        if len(text_lines) < 3:
            # Fallback: try edge-based detection
            text_lines = self._detect_lines_by_structure(gray, params)
        
        if len(text_lines) < 3:
            # Still not enough lines, return original
            return image.copy()
        
        # Build 2D displacement surface
        surface_y, surface_x = self._build_displacement_surface(
            text_lines, w, h, spine_x, params
        )
        self._last_surface = (surface_x, surface_y)
        
        # Create and apply mesh
        map_x, map_y = self._create_adaptive_mesh(
            image.shape, surface_x, surface_y, spine_x, params
        )
        
        # Apply dewarping
        dewarped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return dewarped
    
    def _is_book_spread(self, gray: np.ndarray, 
                       params: EnhancedDewarpParams) -> bool:
        """Detect if image contains a book spread (two facing pages)."""
        if params.is_book_spread is not None:
            return params.is_book_spread
        
        h, w = gray.shape
        
        # For book spread detection, look at the center region of the image
        # and check for a vertical dark line (gutter)
        center_x = w // 2
        search_width = w // 3
        search_left = center_x - search_width // 2
        search_right = center_x + search_width // 2
        
        # Use middle 70% vertically to avoid header/footer areas
        y_margin = int(h * 0.15)
        center_strip = gray[y_margin:h-y_margin, search_left:search_right]
        
        # Compute vertical profile
        vertical_profile = np.mean(center_strip, axis=0)
        
        # Smooth
        vertical_profile = gaussian_filter1d(vertical_profile, sigma=len(vertical_profile)/20)
        
        # Find minimum (darkest column = gutter)
        min_val = np.min(vertical_profile)
        mean_val = np.mean(vertical_profile)
        min_idx = np.argmin(vertical_profile)
        
        # Check aspect ratio - spreads are wider than tall
        # Relaxed threshold for tight crops (e.g. 1.26 aspect ratio)
        if w < h * 1.1:
            return False
        
        # More relaxed threshold - even a 5% dip can indicate a gutter
        if min_val < mean_val * 0.97:
            self._debug_info['spread_detected'] = True
            self._debug_info['gutter_ratio'] = min_val / mean_val
            self._debug_info['gutter_global_x'] = min_idx + search_left
            return True
        
        return False
    
    def _dewarp_book_spread(self, image: np.ndarray, gray: np.ndarray,
                           params: EnhancedDewarpParams) -> np.ndarray:
        """Dewarp a book spread by processing each page separately."""
        h, w = image.shape[:2]
        
        # Find the center spine
        center_x = self._find_spread_center(gray)
        self._spine_x = center_x
        
        # Process left page (spine on right side of this crop)
        left_image = image[:, :center_x + 50]  # Slight overlap
        left_gray = gray[:, :center_x + 50]
        left_dewarped = self._dewarp_single_page(
            left_image, left_gray, 'right', params
        )
        
        # Process right page (spine on left side of this crop)
        right_image = image[:, center_x - 50:]  # Slight overlap
        right_gray = gray[:, center_x - 50:]
        right_dewarped = self._dewarp_single_page(
            right_image, right_gray, 'left', params
        )
        
        # Blend the overlapping region
        result = np.zeros_like(image)
        result[:, :center_x] = left_dewarped[:, :center_x]
        result[:, center_x:] = right_dewarped[:, 50:]
        
        return result
    
    def _find_spread_center(self, gray: np.ndarray) -> int:
        """Find the center gutter of a book spread.
        
        For book spreads, the gutter is ALWAYS very close to the horizontal center
        of the image. We just need to find it within a narrow region.
        """
        h, w = gray.shape
        
        # The gutter must be within 10% of image center
        center_x = w // 2
        search_margin = int(w * 0.10)  # Only search +/- 10%
        search_left = center_x - search_margin
        search_right = center_x + search_margin
        
        # Use middle 70% vertically
        y_margin = int(h * 0.15)
        center_strip = gray[y_margin:h-y_margin, search_left:search_right]
        
        # Find the darkest column in this narrow strip
        brightness = np.mean(center_strip, axis=0)
        brightness = gaussian_filter1d(brightness, sigma=len(brightness)/10)
        
        # Find minimum (darkest = gutter)
        min_idx = np.argmin(brightness)
        
        return min_idx + search_left
    
    def _dewarp_single_page(self, image: np.ndarray, gray: np.ndarray,
                           spine_side: str, 
                           params: EnhancedDewarpParams) -> np.ndarray:
        """Dewarp a single page with known spine position."""
        h, w = image.shape[:2]
        
        # Set spine position
        if spine_side == 'left':
            spine_x = 0
        elif spine_side == 'right':
            spine_x = w
        else:
            spine_x = w // 2
        
        # Detect text lines
        text_lines = self._detect_text_lines_ensemble(gray, params)
        
        if len(text_lines) < 3:
            text_lines = self._detect_lines_by_structure(gray, params)
        
        if len(text_lines) < 3:
            return image.copy()
        
        # Build displacement surface
        surface_y, surface_x = self._build_displacement_surface(
            text_lines, w, h, spine_x, params
        )
        
        # Create mesh
        map_x, map_y = self._create_adaptive_mesh(
            image.shape, surface_x, surface_y, spine_x, params
        )
        
        # Apply
        return cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
    
    def crop_to_content(self, image: np.ndarray, 
                       params: EnhancedDewarpParams) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Crop the image to the detected page content (entire spread).
        
        Uses robust foreground detection to find the book spread against the background.
        This preserves both pages of a spread, unlike quad detection which often matches just one.
        """
        h, w = image.shape[:2]
        
        # Downscale for faster processing
        scale = min(1.0, 1000 / max(h, w))
        small = cv2.resize(image, None, fx=scale, fy=scale)
        sh, sw = small.shape[:2]
        
        # Convert to grayscale
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        else:
            gray = small.copy()
            
        # 1. Use Otsu's thresholding to separate bright pages from dark background
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Morphological closing to fill gaps (like text vs paper)
        kernel_size = int(sw * 0.01)  # 1% of width
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 3. Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, 4, cv2.CV_32S)
        
        # Find likely page components
        # Filter by area (must be > 5% of image)
        min_area = (sh * sw) * 0.05
        
        # Find bounding box covering all large components
        min_x, min_y = sw, sh
        max_x, max_y = 0, 0
        found_content = False
        
        for i in range(1, num_labels):  # Skip background 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w_comp = stats[i, cv2.CC_STAT_WIDTH]
                h_comp = stats[i, cv2.CC_STAT_HEIGHT]
                
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w_comp)
                max_y = max(max_y, y + h_comp)
                found_content = True
                
        if not found_content:
            return None, None
            
        # Scale back up
        min_x = int(min_x / scale)
        min_y = int(min_y / scale)
        max_x = int(max_x / scale)
        max_y = int(max_y / scale)
        
        # Add margin
        margin = params.crop_margin
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(w, max_x + margin)
        max_y = min(h, max_y + margin)
        
        # Check if crop is valid
        if (max_x - min_x) < 100 or (max_y - min_y) < 100:
            return None, None
        
        # Check if we are cropping significantly (avoid trivial crops)
        original_area = h * w
        crop_area = (max_x - min_x) * (max_y - min_y)
        if crop_area > original_area * 0.95:
             # Don't crop if we are keeping almost everything
             return None, None
            
        cropped = image[min_y:max_y, min_x:max_x]
        
        return cropped, {
            'x_offset': min_x,
            'y_offset': min_y,
            'original_width': w,
            'original_height': h
        }
    
    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        """Preprocess image for better detection."""
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=8)
        
        return denoised
    
    def _detect_spine(self, gray: np.ndarray, 
                      params: EnhancedDewarpParams) -> int:
        """
        Detect the spine/gutter position of the book.
        
        The spine is typically the darkest vertical region where
        pages curve the most.
        """
        h, w = gray.shape
        
        if params.spine_position == 'left':
            return 0
        elif params.spine_position == 'right':
            return w
        elif params.spine_position == 'center':
            return w // 2
        
        # Auto-detect: find the darkest vertical strip
        # Compute vertical projection (mean brightness per column)
        # but only in the central region to avoid edges
        margin = w // 10
        central_region = gray[:, margin:w-margin]
        
        vertical_profile = np.mean(central_region, axis=0)
        
        # Smooth the profile
        vertical_profile = gaussian_filter1d(vertical_profile, sigma=w/20)
        
        # Find the minimum (darkest region)
        min_idx = np.argmin(vertical_profile) + margin
        
        # Check if it's actually dark enough to be a spine
        mean_brightness = np.mean(vertical_profile)
        min_brightness = vertical_profile[np.argmin(vertical_profile)]
        
        if min_brightness < mean_brightness * 0.8:
            # There's a distinct dark region - likely the spine
            return min_idx
        
        # No clear spine detected - check if it's a single page
        # by looking at the edges
        left_edge_brightness = np.mean(gray[:, :margin])
        right_edge_brightness = np.mean(gray[:, -margin:])
        
        if left_edge_brightness < right_edge_brightness * 0.8:
            return 0  # Spine on left
        elif right_edge_brightness < left_edge_brightness * 0.8:
            return w  # Spine on right
        
        # Default to center
        return w // 2
    
    def _detect_text_lines_ensemble(self, gray: np.ndarray,
                                    params: EnhancedDewarpParams) -> List[np.ndarray]:
        """
        Detect text lines using multiple methods and combine results.
        """
        all_lines = []
        
        # Method 1: Morphological connected components
        lines1 = self._detect_by_morphology(gray, params)
        all_lines.extend(lines1)
        
        # Method 2: Horizontal projection profile
        lines2 = self._detect_by_projection(gray, params)
        all_lines.extend(lines2)
        
        # Method 3: Hough lines on edges
        lines3 = self._detect_by_hough(gray, params)
        all_lines.extend(lines3)
        
        # Merge overlapping lines
        merged = self._merge_similar_lines(all_lines, gray.shape)
        
        return merged
    
    def _detect_by_morphology(self, gray: np.ndarray,
                              params: EnhancedDewarpParams) -> List[np.ndarray]:
        """Detect text lines using morphological operations."""
        h, w = gray.shape
        
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, 
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Connect text horizontally
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        
        # Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > params.min_line_length and ch < h // 8:
                # Extract centerline
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                points = []
                for xi in range(x, x + cw, 3):
                    col = mask[:, xi]
                    ys = np.where(col > 0)[0]
                    if len(ys) > 0:
                        cy = int(np.mean(ys))
                        points.append([xi, cy])
                
                if len(points) >= 10:
                    lines.append(np.array(points))
        
        return lines
    
    def _detect_by_projection(self, gray: np.ndarray,
                              params: EnhancedDewarpParams) -> List[np.ndarray]:
        """Detect text lines using horizontal projection profile."""
        h, w = gray.shape
        
        # Compute horizontal projection
        projection = np.sum(255 - gray, axis=1).astype(np.float32)
        
        # Smooth
        projection = gaussian_filter1d(projection, sigma=3)
        
        # Find peaks (text line centers)
        threshold = np.mean(projection) + np.std(projection) * 0.5
        
        # Find regions above threshold
        above = projection > threshold
        
        # Find transitions
        diff = np.diff(above.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0 or len(ends) == 0:
            return []
        
        # Align starts and ends
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(ends) < len(starts):
            starts = starts[:len(ends)]
        
        lines = []
        for start, end in zip(starts, ends):
            if end - start < 5:
                continue
            
            # Center of this text line band
            center_y = (start + end) // 2
            
            # Sample points along this horizontal band
            band = gray[max(0, center_y-3):min(h, center_y+3), :]
            
            # Find where there's actually text
            text_presence = np.mean(255 - band, axis=0)
            text_threshold = np.mean(text_presence) * 0.5
            
            points = []
            in_text = False
            text_start = 0
            
            for x in range(w):
                if text_presence[x] > text_threshold:
                    if not in_text:
                        text_start = x
                        in_text = True
                else:
                    if in_text and x - text_start > 20:
                        # Sample this text segment
                        for xi in range(text_start, x, 5):
                            points.append([xi, center_y])
                    in_text = False
            
            if len(points) >= 10:
                lines.append(np.array(points))
        
        return lines
    
    def _detect_by_hough(self, gray: np.ndarray,
                         params: EnhancedDewarpParams) -> List[np.ndarray]:
        """Detect lines using Hough transform."""
        h, w = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Hough lines
        lines_p = cv2.HoughLinesP(
            dilated, rho=1, theta=np.pi/180,
            threshold=30,
            minLineLength=params.min_line_length,
            maxLineGap=params.line_gap
        )
        
        if lines_p is None:
            return []
        
        # Group nearly horizontal lines by y-position
        line_groups = {}
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            # Only keep nearly horizontal lines
            if abs(y2 - y1) < abs(x2 - x1) * 0.3:
                mid_y = (y1 + y2) // 2
                group_y = (mid_y // 15) * 15
                
                if group_y not in line_groups:
                    line_groups[group_y] = []
                line_groups[group_y].append(line[0])
        
        lines = []
        for group_y, group_lines in line_groups.items():
            if len(group_lines) >= 2:
                points = []
                for x1, y1, x2, y2 in group_lines:
                    points.append([x1, y1])
                    points.append([x2, y2])
                points = np.array(sorted(points, key=lambda p: p[0]))
                if len(points) >= 5:
                    lines.append(points)
        
        return lines
    
    def _detect_lines_by_structure(self, gray: np.ndarray,
                                   params: EnhancedDewarpParams) -> List[np.ndarray]:
        """Fallback: detect any horizontal structure in the image."""
        h, w = gray.shape
        
        # Use Sobel to find horizontal edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_abs = np.abs(sobel_x).astype(np.uint8)
        
        # Threshold
        _, binary = cv2.threshold(sobel_abs, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Connect horizontally
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Horizontal projection to find line centers
        proj = np.sum(connected, axis=1)
        
        # Find peaks
        lines = []
        threshold = np.max(proj) * 0.2
        
        in_peak = False
        peak_start = 0
        
        for y in range(h):
            if proj[y] > threshold:
                if not in_peak:
                    peak_start = y
                    in_peak = True
            else:
                if in_peak:
                    center_y = (peak_start + y) // 2
                    # Create a line across the width
                    points = [[x, center_y] for x in range(0, w, 10)]
                    lines.append(np.array(points))
                in_peak = False
        
        return lines
    
    def _merge_similar_lines(self, lines: List[np.ndarray],
                             shape: Tuple[int, int]) -> List[np.ndarray]:
        """Merge lines that are at similar y-positions."""
        if len(lines) == 0:
            return []
        
        h, w = shape
        
        # Group by average y-position
        y_threshold = h / 50
        
        # Sort lines by mean y
        lines_with_y = [(line, np.mean(line[:, 1])) for line in lines]
        lines_with_y.sort(key=lambda x: x[1])
        
        merged = []
        current_group = [lines_with_y[0][0]]
        current_y = lines_with_y[0][1]
        
        for line, y in lines_with_y[1:]:
            if abs(y - current_y) < y_threshold:
                current_group.append(line)
            else:
                # Merge current group
                if current_group:
                    all_points = np.vstack(current_group)
                    # Sort by x and remove duplicates
                    sorted_points = all_points[all_points[:, 0].argsort()]
                    merged.append(sorted_points)
                current_group = [line]
                current_y = y
        
        # Don't forget last group
        if current_group:
            all_points = np.vstack(current_group)
            sorted_points = all_points[all_points[:, 0].argsort()]
            merged.append(sorted_points)
        
        return merged
    
    def _build_displacement_surface(self, text_lines: List[np.ndarray],
                                    width: int, height: int,
                                    spine_x: int,
                                    params: EnhancedDewarpParams
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build 2D displacement surfaces for X and Y corrections.
        
        Uses a cylindrical page model + Keystone correction.
        """
        # 1. Estimate global curvature
        curvature = self._estimate_global_curvature(text_lines, width, height, spine_x)
        
        # 2. Estimate Keystone/Vertical Slope (the "frowning" effect)
        # Positive slope means text goes DOWN as it goes right
        avg_slope = 0.0
        if len(text_lines) > 3:
            slopes = []
            for line in text_lines:
                if len(line) > 10:
                    # Fit line: y = mx + c
                    vx, vy, x0, y0 = cv2.fitLine(line.astype(float), cv2.DIST_L2, 0, 0.01, 0.01)
                    if abs(vx) > 0.01:
                        slopes.append(vy/vx)
            if slopes:
                avg_slope = np.median(slopes)

        # 3. Build Surfaces
        surface_y = np.zeros((height, width), dtype=np.float32)
        surface_x = np.zeros((height, width), dtype=np.float32)
        
        norm_spine_x = spine_x
        
        for x in range(width):
            # Cylindrical component
            dist_from_spine = abs(x - norm_spine_x) / max(norm_spine_x, width - norm_spine_x, 1)
            spine_factor = max(0, 1.0 - dist_from_spine**2)
            
            # Keystone component (linear vertical displacement based on x)
            # If slope is positive (down), we need to shift pixels UP (negative y) to correct
            keystone_y = -(x - norm_spine_x) * avg_slope
            
            for y in range(height):
                # Cylindrical vertical variation
                vert_dist = abs(y - height/2) / (height/2)
                vert_factor = 1.0 - vert_dist * 0.3
                
                cyl_y = curvature * spine_factor * vert_factor
                
                # Combine
                surface_y[y, x] = cyl_y + keystone_y
                
                # Horizontal COMPRESSION near spine (foreshortening correction)
                # Text appears LARGER near spine in curved source, so we SHRINK it
                # by sampling from a WIDER source range (surface_x positive for x > spine)
                compress = 0.15 * spine_factor  # Compression factor
                surface_x[y, x] = compress * (x - norm_spine_x)

        return surface_y, surface_x

    def _estimate_global_curvature(self, text_lines: List[np.ndarray],
                                   width: int, height: int,
                                   spine_x: int) -> float:
        """Estimate global curvature magnitude."""
        if len(text_lines) < 3:
            return 20.0
            
        curvatures = []
        for line_points in text_lines:
            if len(line_points) < 10: continue
            try:
                z = np.polyfit(line_points[:,0], line_points[:,1], 2)
                curvatures.append(abs(z[0]) * (width/2)**2)
            except: pass
            
        if curvatures:
            return np.median(curvatures)
        return 20.0
    
    def _interpolate_surface(self, deviations: np.ndarray,
                             width: int, height: int,
                             spine_x: int,
                             params: EnhancedDewarpParams,
                             direction: str) -> np.ndarray:
        """Interpolate deviation points into a smooth surface."""
        # Bin deviations into a grid
        n_bins_x = params.num_control_points_x
        n_bins_y = params.num_control_points_y
        
        bin_w = width / n_bins_x
        bin_h = height / n_bins_y
        
        # Create grid for binned values
        grid = np.zeros((n_bins_y, n_bins_x))
        counts = np.zeros((n_bins_y, n_bins_x))
        
        for x, y, dev in deviations:
            bx = min(int(x / bin_w), n_bins_x - 1)
            by = min(int(y / bin_h), n_bins_y - 1)
            grid[by, bx] += dev
            counts[by, bx] += 1
        
        # Average
        with np.errstate(divide='ignore', invalid='ignore'):
            grid = np.where(counts > 0, grid / counts, 0)
        
        # Fill holes by interpolation
        grid = self._fill_holes(grid)
        
        # Smooth
        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=params.smoothing_factor * 3)
        
        # Interpolate to full resolution
        x_coords = np.linspace(0, width - 1, n_bins_x)
        y_coords = np.linspace(0, height - 1, n_bins_y)
        
        spline = RectBivariateSpline(y_coords, x_coords, grid, kx=3, ky=3)
        
        full_x = np.arange(width)
        full_y = np.arange(height)
        surface = spline(full_y, full_x)
        
        return surface
    
    def _build_x_surface(self, slope_data: List[List[float]],
                         width: int, height: int,
                         spine_x: int,
                         params: EnhancedDewarpParams) -> np.ndarray:
        """
        Build X displacement surface based on line slopes.
        
        Near the spine, text lines curve which causes horizontal
        compression that we need to correct.
        """
        # Simple model: X displacement increases near spine
        surface = np.zeros((height, width))
        
        if len(slope_data) < 3:
            return surface
        
        # Estimate curvature from slopes
        slopes = np.array([s[1] for s in slope_data])
        mean_slope = np.mean(slopes)
        slope_variation = np.std(slopes)
        
        # If there's significant slope variation, there's curvature
        if slope_variation < 0.001:
            return surface
        
        # Create horizontal stretch pattern
        # Maximum stretch at spine, decreasing outward
        for x in range(width):
            # Distance from spine (normalized 0-1)
            dist_from_spine = abs(x - spine_x) / max(spine_x, width - spine_x)
            
            # Stretch factor diminishes with distance from spine
            stretch_factor = np.exp(-dist_from_spine / params.spine_falloff)
            stretch_factor = max(0, stretch_factor - 0.1)  # Threshold
            
            # Direction of stretch depends on which side of spine
            direction = 1 if x < spine_x else -1
            
            for y in range(height):
                # Vary stretch with vertical position too
                vert_factor = 1.0 - 0.3 * abs(y - height/2) / (height/2)
                surface[y, x] = direction * stretch_factor * vert_factor * params.horizontal_strength * 10
        
        return surface
    
    def _fill_holes(self, grid: np.ndarray) -> np.ndarray:
        """Fill zero values in grid by nearest neighbor interpolation."""
        from scipy.ndimage import distance_transform_edt
        
        mask = grid == 0
        if not np.any(mask):
            return grid
        
        # Find nearest non-zero value
        indices = distance_transform_edt(mask, return_distances=False, 
                                         return_indices=True)
        filled = grid[tuple(indices)]
        
        return filled
    
    def _create_adaptive_mesh(self, shape: Tuple[int, ...],
                              surface_x: np.ndarray,
                              surface_y: np.ndarray,
                              spine_x: int,
                              params: EnhancedDewarpParams
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create mapping arrays with adaptive density near spine.
        """
        h, w = shape[:2]
        
        # Base mesh
        map_y, map_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # Ensure surfaces are the right shape and type
        surface_y = surface_y.astype(np.float32)
        surface_x = surface_x.astype(np.float32)
        
        # Compute correction factors vectorized for speed
        y_indices = np.arange(h, dtype=np.float32).reshape(-1, 1)
        x_indices = np.arange(w, dtype=np.float32).reshape(1, -1)
        
        # Vertical factor: full correction in middle, reduced at top/bottom
        vert_factor = 1.0 - 0.3 * np.abs(y_indices - h/2) / (h/2)
        vert_factor = np.maximum(0.5, vert_factor).astype(np.float32)
        
        # Spine factor: stronger correction near spine
        max_dist = max(spine_x, w - spine_x, 1)
        dist_from_spine = np.abs(x_indices - spine_x) / max_dist
        spine_factor = np.exp(-dist_from_spine / params.spine_falloff).astype(np.float32)
        
        # Combined correction mask
        correction_mask = (vert_factor * spine_factor).astype(np.float32)
        
        # Apply corrections
        map_y = map_y + (surface_y * correction_mask * params.vertical_strength).astype(np.float32)
        map_x = map_x + (surface_x * params.horizontal_strength).astype(np.float32)
        
        return map_x, map_y
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information from last dewarp operation."""
        return {
            'spine_x': self._spine_x,
            'last_surface': self._last_surface,
            **self._debug_info
        }
    
    def get_preview(self, image: np.ndarray) -> np.ndarray:
        """Create a preview showing detected features."""
        preview = image.copy()
        h, w = preview.shape[:2]
        
        # Draw spine line
        spine_x = self._spine_x
        
        # Adjust spine_x if image was cropped
        if 'crop_info' in self._debug_info and spine_x is not None:
             # Check if dimensions match original (preview is usually original)
             orig_w = self._debug_info['crop_info']['original_width']
             if abs(w - orig_w) < 5:
                 offset_x = self._debug_info['crop_info']['x_offset']
                 spine_x += offset_x
        
        if spine_x is not None:
            cv2.line(preview, (int(spine_x), 0), (int(spine_x), h), 
                    (0, 0, 255), 3)  # Red line
        
        return preview
