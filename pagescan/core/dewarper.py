"""
Page dewarping module for correcting curved text on book pages.

Uses mesh-based dewarping to flatten curved text lines, particularly
useful for book pages photographed near the spine/gutter.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class DewarpParams:
    """Parameters for the dewarping algorithm."""
    # Detection parameters
    text_threshold: int = 127  # Threshold for text detection
    min_line_length: int = 100  # Minimum length for detected lines
    line_gap: int = 10  # Maximum gap between line segments

    # Curve fitting parameters
    num_control_points: int = 20  # Number of control points along curve
    smoothing_factor: float = 0.5  # Spline smoothing (0=interpolate, higher=smoother)

    # Mesh parameters
    mesh_rows: int = 50  # Number of rows in dewarping mesh
    mesh_cols: int = 50  # Number of columns in dewarping mesh

    # Manual adjustment
    manual_curve_points: Optional[List[Tuple[int, int]]] = None


class PageDewarper:
    """Handles dewarping of curved book pages."""

    def __init__(self, params: Optional[DewarpParams] = None):
        self.params = params or DewarpParams()
        self._last_curve = None
        self._last_mesh = None

    def dewarp(self, image: np.ndarray, params: Optional[DewarpParams] = None) -> np.ndarray:
        """
        Dewarp a curved book page.

        Args:
            image: Input image as RGB numpy array
            params: Optional parameters override

        Returns:
            Dewarped image
        """
        params = params or self.params

        # If manual curve points provided, use them
        if params.manual_curve_points and len(params.manual_curve_points) >= 3:
            curve = self._fit_manual_curve(params.manual_curve_points, image.shape[1])
        else:
            # Auto-detect curve
            curve = self._detect_curve(image, params)

        if curve is None:
            # No curve detected, return original
            return image.copy()

        self._last_curve = curve

        # Create dewarping mesh
        mesh = self._create_dewarp_mesh(image.shape, curve, params)
        self._last_mesh = mesh

        # Apply dewarping
        return self._apply_mesh(image, mesh)

    def _detect_curve(self, image: np.ndarray, params: DewarpParams) -> Optional[np.ndarray]:
        """
        Detect the curve of text lines in the image.

        Returns:
            Array of y-offsets for each x position, or None if detection fails
        """
        h, w = image.shape[:2]

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Enhance text for better detection
        gray = self._preprocess_for_detection(gray)

        # Detect text lines using horizontal projection
        text_lines = self._detect_text_lines(gray, params)

        if len(text_lines) < 3:
            # Try alternative method with edge detection
            text_lines = self._detect_lines_by_edges(gray, params)

        if len(text_lines) < 3:
            return None

        # Fit curve through detected lines
        curve = self._fit_curve_through_lines(text_lines, w, h, params)

        return curve

    def _preprocess_for_detection(self, gray: np.ndarray) -> np.ndarray:
        """Preprocess grayscale image for text detection."""
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        return denoised

    def _detect_text_lines(self, gray: np.ndarray, params: DewarpParams) -> List[np.ndarray]:
        """
        Detect text lines using horizontal projection profiles.

        Returns:
            List of arrays, each containing (x, y) points for a detected line
        """
        h, w = gray.shape

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations to connect text
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)

        # Find contours of text regions
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and sort contours by y-position
        lines = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > params.min_line_length and ch < h // 10:  # Filter by size
                # Get the midline of this text region
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)

                # Find the center y for each x
                line_points = []
                for xi in range(x, x + cw, 5):  # Sample every 5 pixels
                    col = mask[:, xi]
                    ys = np.where(col > 0)[0]
                    if len(ys) > 0:
                        cy = int(np.mean(ys))
                        line_points.append([xi, cy])

                if len(line_points) >= 10:
                    lines.append(np.array(line_points))

        return lines

    def _detect_lines_by_edges(self, gray: np.ndarray, params: DewarpParams) -> List[np.ndarray]:
        """
        Alternative line detection using Hough transform.
        """
        h, w = gray.shape

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Use probabilistic Hough transform
        lines_p = cv2.HoughLinesP(
            dilated,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=params.min_line_length,
            maxLineGap=params.line_gap
        )

        if lines_p is None:
            return []

        # Group lines by y-position
        line_groups = {}
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            mid_y = (y1 + y2) // 2
            # Round to nearest 20 pixels for grouping
            group_y = (mid_y // 20) * 20

            if group_y not in line_groups:
                line_groups[group_y] = []
            line_groups[group_y].append(line[0])

        # Convert groups to line arrays
        lines = []
        for group_y, group_lines in line_groups.items():
            if len(group_lines) >= 3:
                # Combine lines in group
                points = []
                for x1, y1, x2, y2 in group_lines:
                    points.append([x1, y1])
                    points.append([x2, y2])
                points = np.array(sorted(points, key=lambda p: p[0]))
                if len(points) >= 5:
                    lines.append(points)

        return lines

    def _fit_curve_through_lines(self, text_lines: List[np.ndarray],
                                  width: int, height: int,
                                  params: DewarpParams) -> np.ndarray:
        """
        Fit a smooth curve through detected text lines.

        Returns:
            Array of y-offsets for each x position
        """
        # Collect all curve deviations
        deviations = []

        for line_points in text_lines:
            if len(line_points) < 5:
                continue

            xs = line_points[:, 0]
            ys = line_points[:, 1]

            # Fit a linear baseline
            z = np.polyfit(xs, ys, 1)
            baseline = np.polyval(z, xs)

            # Calculate deviation from baseline
            for x, y, bl in zip(xs, ys, baseline):
                deviations.append([x, y - bl])

        if len(deviations) < 10:
            return np.zeros(width)

        deviations = np.array(deviations)

        # Bin deviations by x-position and average
        num_bins = params.num_control_points
        bin_width = width / num_bins
        binned_x = []
        binned_dev = []

        for i in range(num_bins):
            x_min = i * bin_width
            x_max = (i + 1) * bin_width
            mask = (deviations[:, 0] >= x_min) & (deviations[:, 0] < x_max)
            if np.any(mask):
                binned_x.append((x_min + x_max) / 2)
                binned_dev.append(np.median(deviations[mask, 1]))

        if len(binned_x) < 3:
            return np.zeros(width)

        binned_x = np.array(binned_x)
        binned_dev = np.array(binned_dev)

        # Smooth the curve
        if len(binned_x) >= 5:
            # Use Savitzky-Golay filter for smoothing
            window = min(len(binned_dev), 7)
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                binned_dev = savgol_filter(binned_dev, window, 2)

        # Interpolate to full width
        interp_func = interp1d(binned_x, binned_dev, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        full_curve = interp_func(np.arange(width))

        # Remove any extreme values
        std = np.std(full_curve)
        mean = np.mean(full_curve)
        full_curve = np.clip(full_curve, mean - 3 * std, mean + 3 * std)

        return full_curve

    def _fit_manual_curve(self, points: List[Tuple[int, int]], width: int) -> np.ndarray:
        """
        Fit a curve through manually specified control points.
        """
        points = sorted(points, key=lambda p: p[0])
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        # Find the mean y to calculate deviations
        mean_y = np.mean(ys)
        devs = ys - mean_y

        # Interpolate
        if len(points) >= 4:
            interp_func = interp1d(xs, devs, kind='cubic',
                                   bounds_error=False, fill_value='extrapolate')
        else:
            interp_func = interp1d(xs, devs, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')

        return interp_func(np.arange(width))

    def _create_dewarp_mesh(self, shape: Tuple[int, ...], curve: np.ndarray,
                            params: DewarpParams) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a mesh for dewarping based on the detected curve.

        Returns:
            Tuple of (map_x, map_y) arrays for cv2.remap
        """
        h, w = shape[:2]
        rows = params.mesh_rows
        cols = params.mesh_cols

        # Create base mesh grid
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        # Normalize curve to proportion of height
        max_dev = np.max(np.abs(curve))
        if max_dev < 1:
            # No significant curve, return identity mapping
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            return map_x.astype(np.float32), map_y.astype(np.float32)

        # Scale factor - how much to correct
        # The curve represents the vertical deviation, we want to "undo" it
        scale = min(1.0, h / (4 * max_dev))  # Limit correction to avoid artifacts

        for y in range(h):
            for x in range(w):
                # Calculate the amount of curve correction at this position
                # More correction at the middle of the page, less at edges
                vertical_factor = 1.0 - abs(y - h / 2) / (h / 2)
                vertical_factor = max(0, vertical_factor)

                correction = curve[x] * vertical_factor * scale

                # Map to source coordinates
                map_x[y, x] = x
                map_y[y, x] = y + correction

        return map_x, map_y

    def _apply_mesh(self, image: np.ndarray,
                    mesh: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Apply the dewarping mesh to the image."""
        map_x, map_y = mesh

        # Use cv2.remap for efficient image transformation
        dewarped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return dewarped

    def get_last_curve(self) -> Optional[np.ndarray]:
        """Get the last detected/used curve."""
        return self._last_curve

    def get_curve_preview(self, image: np.ndarray,
                          curve: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a preview image showing the detected curve overlay.

        Args:
            image: Input image
            curve: Curve to display (uses last detected if None)

        Returns:
            Image with curve overlay
        """
        if curve is None:
            curve = self._last_curve

        if curve is None:
            return image.copy()

        preview = image.copy()
        h, w = preview.shape[:2]

        # Draw the curve at the center of the image
        center_y = h // 2
        points = []
        for x in range(0, w, 5):
            y = int(center_y + curve[x])
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        # Draw the curve line
        cv2.polylines(preview, [points], False, (255, 0, 0), 2)

        # Draw horizontal reference line
        cv2.line(preview, (0, center_y), (w, center_y), (0, 255, 0), 1)

        return preview

    def analyze_curve_severity(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the severity of page curvature.

        Returns:
            Dictionary with curve analysis metrics
        """
        curve = self._detect_curve(image, self.params)

        if curve is None:
            return {
                'detected': False,
                'severity': 'none',
                'max_deviation': 0,
                'curve_direction': 'flat'
            }

        max_dev = np.max(np.abs(curve))
        mean_curve = np.mean(curve)

        # Determine severity
        h = image.shape[0]
        dev_ratio = max_dev / h

        if dev_ratio < 0.01:
            severity = 'minimal'
        elif dev_ratio < 0.03:
            severity = 'low'
        elif dev_ratio < 0.06:
            severity = 'medium'
        else:
            severity = 'high'

        # Determine direction
        if mean_curve > 5:
            direction = 'convex'  # Bulging up
        elif mean_curve < -5:
            direction = 'concave'  # Curving down
        else:
            direction = 'symmetric'

        return {
            'detected': True,
            'severity': severity,
            'max_deviation': float(max_dev),
            'mean_deviation': float(mean_curve),
            'curve_direction': direction,
            'deviation_ratio': float(dev_ratio)
        }


class AdvancedDewarper(PageDewarper):
    """
    Advanced dewarper with additional algorithms for difficult cases.
    """

    def __init__(self, params: Optional[DewarpParams] = None):
        super().__init__(params)

    def dewarp_with_grid(self, image: np.ndarray,
                         grid_points: np.ndarray) -> np.ndarray:
        """
        Dewarp using a manually specified grid of control points.

        Args:
            image: Input image
            grid_points: Array of shape (rows, cols, 2) with (x, y) coordinates

        Returns:
            Dewarped image
        """
        h, w = image.shape[:2]
        rows, cols, _ = grid_points.shape

        # Create destination points (regular grid)
        dst_points = np.zeros_like(grid_points)
        for i in range(rows):
            for j in range(cols):
                dst_points[i, j] = [j * w / (cols - 1), i * h / (rows - 1)]

        # Create the mapping using piecewise affine transform
        map_x, map_y = self._create_piecewise_map(grid_points, dst_points, (h, w))

        return cv2.remap(image, map_x, map_y,
                        interpolation=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)

    def _create_piecewise_map(self, src_points: np.ndarray,
                               dst_points: np.ndarray,
                               shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create mapping arrays from control point grid."""
        h, w = shape
        rows, cols, _ = src_points.shape

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        # For each cell in the grid, compute local transform
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Get the four corners of this cell
                src_quad = np.array([
                    src_points[i, j],
                    src_points[i, j + 1],
                    src_points[i + 1, j + 1],
                    src_points[i + 1, j]
                ], dtype=np.float32)

                dst_quad = np.array([
                    dst_points[i, j],
                    dst_points[i, j + 1],
                    dst_points[i + 1, j + 1],
                    dst_points[i + 1, j]
                ], dtype=np.float32)

                # Compute perspective transform for this cell
                M = cv2.getPerspectiveTransform(dst_quad, src_quad)

                # Apply to all pixels in the destination cell region
                x_min = int(dst_quad[:, 0].min())
                x_max = int(dst_quad[:, 0].max()) + 1
                y_min = int(dst_quad[:, 1].min())
                y_max = int(dst_quad[:, 1].max()) + 1

                # Clip to image bounds
                x_min = max(0, x_min)
                x_max = min(w, x_max)
                y_min = max(0, y_min)
                y_max = min(h, y_max)

                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        # Check if point is inside destination quad
                        if self._point_in_quad([x, y], dst_quad):
                            # Transform point
                            pt = np.array([[[x, y]]], dtype=np.float32)
                            transformed = cv2.perspectiveTransform(pt, M)
                            map_x[y, x] = transformed[0, 0, 0]
                            map_y[y, x] = transformed[0, 0, 1]

        return map_x, map_y

    def _point_in_quad(self, point: List[int], quad: np.ndarray) -> bool:
        """Check if a point is inside a quadrilateral."""
        return cv2.pointPolygonTest(quad.astype(np.float32),
                                    (float(point[0]), float(point[1])), False) >= 0
