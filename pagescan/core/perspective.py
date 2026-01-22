"""
Perspective correction module for fixing keystone/trapezoidal distortion.

Provides automatic detection of page boundaries and perspective transformation
to rectangular output.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class PerspectiveParams:
    """Parameters for perspective correction."""
    # Edge detection
    canny_low: int = 50
    canny_high: int = 150

    # Contour filtering
    min_area_ratio: float = 0.1  # Minimum area as ratio of image
    max_area_ratio: float = 0.98  # Maximum area as ratio of image

    # Output dimensions
    output_width: Optional[int] = None  # None = auto from detected quad
    output_height: Optional[int] = None

    # Margin to add around detected page
    margin: int = 0


class PerspectiveCorrector:
    """Handles perspective/keystone correction for scanned pages."""

    def __init__(self, params: Optional[PerspectiveParams] = None):
        self.params = params or PerspectiveParams()
        self._last_quad = None
        self._last_output_size = None

    def correct(self, image: np.ndarray,
                corners: Optional[np.ndarray] = None,
                params: Optional[PerspectiveParams] = None) -> np.ndarray:
        """
        Apply perspective correction to an image.

        Args:
            image: Input image as RGB numpy array
            corners: Optional manual corner points as shape (4, 2) array
                    Order: top-left, top-right, bottom-right, bottom-left
            params: Optional parameters override

        Returns:
            Perspective-corrected image
        """
        params = params or self.params

        if corners is None:
            corners = self.detect_page_corners(image, params)

        if corners is None:
            # No page detected, return original
            return image.copy()

        self._last_quad = corners

        # Determine output size
        output_size = self._calculate_output_size(corners, params)
        self._last_output_size = output_size

        # Create destination points (rectangle)
        dst_points = np.array([
            [params.margin, params.margin],
            [output_size[0] - params.margin, params.margin],
            [output_size[0] - params.margin, output_size[1] - params.margin],
            [params.margin, output_size[1] - params.margin]
        ], dtype=np.float32)

        # Get perspective transform matrix
        src_points = corners.astype(np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply transform
        corrected = cv2.warpPerspective(
            image, M, output_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return corrected

    def detect_page_corners(self, image: np.ndarray,
                            params: Optional[PerspectiveParams] = None) -> Optional[np.ndarray]:
        """
        Automatically detect the four corners of the page.

        Returns:
            Array of shape (4, 2) with corner coordinates in order:
            top-left, top-right, bottom-right, bottom-left
            Or None if detection fails
        """
        params = params or self.params
        h, w = image.shape[:2]

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Try multiple detection methods
        corners = self._detect_by_contours(gray, params)

        if corners is None:
            corners = self._detect_by_lines(gray, params)

        if corners is None:
            corners = self._detect_by_edges(gray, params)

        return corners

    def _detect_by_contours(self, gray: np.ndarray,
                            params: PerspectiveParams) -> Optional[np.ndarray]:
        """Detect page by finding the largest quadrilateral contour."""
        h, w = gray.shape

        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, params.canny_low, params.canny_high)

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter and sort contours by area
        min_area = h * w * params.min_area_ratio
        max_area = h * w * params.max_area_ratio

        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                valid_contours.append((cnt, area))

        if not valid_contours:
            return None

        # Sort by area (largest first)
        valid_contours.sort(key=lambda x: x[1], reverse=True)

        # Try to find a quadrilateral
        for cnt, area in valid_contours:
            # Approximate contour to polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                # Verify it's a valid quadrilateral
                if self._is_valid_quad(corners, (h, w)):
                    return self._order_corners(corners)

        # If no 4-point polygon found, try convex hull + corner detection
        largest_cnt = valid_contours[0][0]
        hull = cv2.convexHull(largest_cnt)

        # Approximate hull to get 4 corners
        peri = cv2.arcLength(hull, True)
        for eps in [0.02, 0.03, 0.04, 0.05, 0.1]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                if self._is_valid_quad(corners, (h, w)):
                    return self._order_corners(corners)

        return None

    def _detect_by_lines(self, gray: np.ndarray,
                         params: PerspectiveParams) -> Optional[np.ndarray]:
        """Detect page corners by finding intersecting lines."""
        h, w = gray.shape

        # Edge detection
        edges = cv2.Canny(gray, params.canny_low, params.canny_high)

        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) < 4:
            return None

        # Separate lines into horizontal and vertical groups
        horizontal = []
        vertical = []

        for line in lines:
            rho, theta = line[0]
            # Horizontal lines: theta near 0 or pi
            if theta < np.pi / 6 or theta > 5 * np.pi / 6:
                vertical.append((rho, theta))
            # Vertical lines: theta near pi/2
            elif np.pi / 3 < theta < 2 * np.pi / 3:
                horizontal.append((rho, theta))

        if len(horizontal) < 2 or len(vertical) < 2:
            return None

        # Get the extreme lines
        horizontal.sort(key=lambda x: x[0])
        vertical.sort(key=lambda x: x[0])

        top_line = horizontal[0]
        bottom_line = horizontal[-1]
        left_line = vertical[0]
        right_line = vertical[-1]

        # Find intersections
        corners = []
        for h_line in [top_line, bottom_line]:
            for v_line in [left_line, right_line]:
                intersection = self._line_intersection(h_line, v_line)
                if intersection is not None:
                    corners.append(intersection)

        if len(corners) != 4:
            return None

        corners = np.array(corners)
        if self._is_valid_quad(corners, (h, w)):
            return self._order_corners(corners)

        return None

    def _detect_by_edges(self, gray: np.ndarray,
                         params: PerspectiveParams) -> Optional[np.ndarray]:
        """Detect page by analyzing edge density in different regions."""
        h, w = gray.shape

        # Use adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find the bounding rectangle of non-zero pixels
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None

        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        if self._is_valid_quad(box, (h, w)):
            return self._order_corners(box)

        return None

    def _line_intersection(self, line1: Tuple[float, float],
                           line2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find intersection of two lines in Hough (rho, theta) format."""
        rho1, theta1 = line1
        rho2, theta2 = line2

        # Convert to ax + by = c form
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)

        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            return None

        x = (b2 * rho1 - b1 * rho2) / det
        y = (a1 * rho2 - a2 * rho1) / det

        return (x, y)

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left.
        """
        # Calculate centroid
        center = corners.mean(axis=0)

        # Sort by angle from center
        angles = np.arctan2(corners[:, 1] - center[1],
                          corners[:, 0] - center[0])

        # Get indices that would sort by angle
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]

        # Find top-left (smallest x + y sum)
        sums = sorted_corners[:, 0] + sorted_corners[:, 1]
        tl_idx = np.argmin(sums)

        # Rotate array so top-left is first
        ordered = np.roll(sorted_corners, -tl_idx, axis=0)

        return ordered

    def _is_valid_quad(self, corners: np.ndarray,
                       image_shape: Tuple[int, int]) -> bool:
        """Check if detected quadrilateral is valid."""
        h, w = image_shape

        # Check all corners are within image bounds (with tolerance)
        margin = max(w, h) * 0.1
        for x, y in corners:
            if x < -margin or x > w + margin or y < -margin or y > h + margin:
                return False

        # Check area is reasonable
        area = cv2.contourArea(corners.astype(np.float32))
        image_area = h * w

        if area < image_area * 0.05 or area > image_area * 1.1:
            return False

        # Check convexity
        hull = cv2.convexHull(corners.astype(np.float32))
        if len(hull) != 4:
            return False

        return True

    def _calculate_output_size(self, corners: np.ndarray,
                               params: PerspectiveParams) -> Tuple[int, int]:
        """Calculate output image dimensions based on detected quad."""
        if params.output_width and params.output_height:
            return (params.output_width, params.output_height)

        # Calculate dimensions from quadrilateral
        # Top edge
        top_width = np.linalg.norm(corners[1] - corners[0])
        # Bottom edge
        bottom_width = np.linalg.norm(corners[2] - corners[3])
        # Left edge
        left_height = np.linalg.norm(corners[3] - corners[0])
        # Right edge
        right_height = np.linalg.norm(corners[2] - corners[1])

        # Use maximum of opposing edges
        width = int(max(top_width, bottom_width))
        height = int(max(left_height, right_height))

        # Add margin
        width += 2 * params.margin
        height += 2 * params.margin

        return (width, height)

    def get_last_quad(self) -> Optional[np.ndarray]:
        """Get the last detected/used quadrilateral."""
        return self._last_quad

    def get_corner_preview(self, image: np.ndarray,
                           corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a preview image showing the detected corners.

        Args:
            image: Input image
            corners: Corner points (uses last detected if None)

        Returns:
            Image with corner overlay
        """
        if corners is None:
            corners = self._last_quad

        preview = image.copy()

        if corners is None:
            return preview

        corners_int = corners.astype(np.int32)

        # Draw quadrilateral
        cv2.polylines(preview, [corners_int], True, (0, 255, 0), 2)

        # Draw corners with labels
        labels = ['TL', 'TR', 'BR', 'BL']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for i, (corner, label, color) in enumerate(zip(corners_int, labels, colors)):
            x, y = corner
            # Draw circle at corner
            cv2.circle(preview, (x, y), 10, color, -1)
            cv2.circle(preview, (x, y), 12, (255, 255, 255), 2)
            # Draw label
            cv2.putText(preview, label, (x + 15, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return preview

    def refine_corners(self, gray: np.ndarray,
                       corners: np.ndarray,
                       window_size: int = 20) -> np.ndarray:
        """
        Refine corner positions using sub-pixel accuracy.

        Args:
            gray: Grayscale image
            corners: Initial corner estimates
            window_size: Search window size

        Returns:
            Refined corner positions
        """
        corners_float = corners.astype(np.float32).reshape(-1, 1, 2)

        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        refined = cv2.cornerSubPix(
            gray, corners_float,
            (window_size, window_size),
            (-1, -1),
            criteria
        )

        return refined.reshape(-1, 2)

    def estimate_perspective_distortion(self, corners: np.ndarray) -> dict:
        """
        Estimate the amount of perspective distortion.

        Returns:
            Dictionary with distortion metrics
        """
        if corners is None:
            return {'distortion': 0, 'type': 'none'}

        # Calculate edge lengths
        top = np.linalg.norm(corners[1] - corners[0])
        bottom = np.linalg.norm(corners[2] - corners[3])
        left = np.linalg.norm(corners[3] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])

        # Calculate ratios
        h_ratio = min(top, bottom) / max(top, bottom) if max(top, bottom) > 0 else 1
        v_ratio = min(left, right) / max(left, right) if max(left, right) > 0 else 1

        # Overall distortion (1 = no distortion, 0 = maximum)
        distortion = 1 - min(h_ratio, v_ratio)

        # Determine type
        if distortion < 0.02:
            dist_type = 'minimal'
        elif h_ratio < v_ratio:
            dist_type = 'horizontal_keystone'
        else:
            dist_type = 'vertical_keystone'

        return {
            'distortion': float(distortion),
            'type': dist_type,
            'h_ratio': float(h_ratio),
            'v_ratio': float(v_ratio),
            'edge_lengths': {
                'top': float(top),
                'bottom': float(bottom),
                'left': float(left),
                'right': float(right)
            }
        }
