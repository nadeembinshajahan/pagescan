"""
Main image processor combining all correction and enhancement features.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum, auto

from .dewarper import PageDewarper, DewarpParams
from .perspective import PerspectiveCorrector, PerspectiveParams


class ProcessingStep(Enum):
    """Available processing steps."""
    DEWARP = auto()
    PERSPECTIVE = auto()
    DESKEW = auto()
    CROP = auto()
    WHITE_BALANCE = auto()
    EXPOSURE = auto()
    BINARIZE = auto()


@dataclass
class ProcessingSettings:
    """Settings for all processing operations."""
    # Enable/disable steps
    enable_dewarp: bool = True
    enable_perspective: bool = True
    enable_deskew: bool = True
    enable_crop: bool = True
    enable_white_balance: bool = False
    enable_exposure: bool = False
    enable_binarize: bool = False

    # Dewarp settings
    dewarp_params: DewarpParams = field(default_factory=DewarpParams)

    # Perspective settings
    perspective_params: PerspectiveParams = field(default_factory=PerspectiveParams)
    manual_corners: Optional[np.ndarray] = None

    # Deskew settings
    deskew_angle: Optional[float] = None  # None = auto-detect

    # Crop settings
    crop_margin: int = 10  # Pixels to leave as margin
    auto_crop: bool = True

    # White balance settings
    wb_method: str = 'gray_world'  # 'gray_world', 'white_patch', 'manual'
    wb_temperature: float = 1.0  # For manual adjustment

    # Exposure settings
    exposure_adjustment: float = 0.0  # EV adjustment (-2 to +2)
    contrast: float = 1.0  # Contrast multiplier
    brightness: int = 0  # Brightness offset

    # Binarization settings
    binarize_method: str = 'otsu'  # 'otsu', 'adaptive', 'sauvola'
    binarize_threshold: int = 127  # For manual threshold
    adaptive_block_size: int = 11
    adaptive_c: int = 2


class ImageProcessor:
    """Main processor combining all image correction features."""

    def __init__(self, settings: Optional[ProcessingSettings] = None):
        self.settings = settings or ProcessingSettings()
        self.dewarper = PageDewarper(self.settings.dewarp_params)
        self.perspective_corrector = PerspectiveCorrector(self.settings.perspective_params)

        self._processing_history: List[Tuple[str, np.ndarray]] = []
        self._original_image: Optional[np.ndarray] = None

    def process(self, image: np.ndarray,
                settings: Optional[ProcessingSettings] = None) -> np.ndarray:
        """
        Apply all enabled processing steps to an image.

        Args:
            image: Input image as RGB numpy array
            settings: Optional settings override

        Returns:
            Processed image
        """
        settings = settings or self.settings
        self._original_image = image.copy()
        self._processing_history = [('original', image.copy())]

        result = image.copy()

        # Apply processing steps in order
        if settings.enable_dewarp:
            result = self._apply_dewarp(result, settings)
            self._processing_history.append(('dewarp', result.copy()))

        if settings.enable_perspective:
            result = self._apply_perspective(result, settings)
            self._processing_history.append(('perspective', result.copy()))

        if settings.enable_deskew:
            result = self._apply_deskew(result, settings)
            self._processing_history.append(('deskew', result.copy()))

        if settings.enable_crop:
            result = self._apply_crop(result, settings)
            self._processing_history.append(('crop', result.copy()))

        if settings.enable_white_balance:
            result = self._apply_white_balance(result, settings)
            self._processing_history.append(('white_balance', result.copy()))

        if settings.enable_exposure:
            result = self._apply_exposure(result, settings)
            self._processing_history.append(('exposure', result.copy()))

        if settings.enable_binarize:
            result = self._apply_binarize(result, settings)
            self._processing_history.append(('binarize', result.copy()))

        return result

    def _apply_dewarp(self, image: np.ndarray,
                      settings: ProcessingSettings) -> np.ndarray:
        """Apply page dewarping."""
        return self.dewarper.dewarp(image, settings.dewarp_params)

    def _apply_perspective(self, image: np.ndarray,
                           settings: ProcessingSettings) -> np.ndarray:
        """Apply perspective correction."""
        return self.perspective_corrector.correct(
            image,
            corners=settings.manual_corners,
            params=settings.perspective_params
        )

    def _apply_deskew(self, image: np.ndarray,
                      settings: ProcessingSettings) -> np.ndarray:
        """Apply rotation to correct skew."""
        if settings.deskew_angle is not None:
            angle = settings.deskew_angle
        else:
            angle = self._detect_skew_angle(image)

        if abs(angle) < 0.1:
            return image

        return self._rotate_image(image, angle)

    def _detect_skew_angle(self, image: np.ndarray) -> float:
        """Detect the skew angle of text in the image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find coordinates of text pixels
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) < 100:
            return 0.0

        # Use PCA to find main text direction
        # Or use Hough transform for more robust detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return 0.0

        # Collect angles near horizontal
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert to degrees from horizontal
            angle = (theta * 180 / np.pi) - 90
            if -15 < angle < 15:  # Only consider near-horizontal lines
                angles.append(angle)

        if not angles:
            return 0.0

        # Return median angle
        return float(np.median(angles))

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle (in degrees)."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box size
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust rotation matrix for new center
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # Apply rotation
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    def _apply_crop(self, image: np.ndarray,
                    settings: ProcessingSettings) -> np.ndarray:
        """Apply auto-crop to remove borders."""
        if not settings.auto_crop:
            return image

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Find content area
        # Use adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 10
        )

        # Find bounding box of content
        coords = cv2.findNonZero(binary)
        if coords is None:
            return image

        x, y, w, h = cv2.boundingRect(coords)

        # Add margin
        margin = settings.crop_margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y:y+h, x:x+w]

    def _apply_white_balance(self, image: np.ndarray,
                              settings: ProcessingSettings) -> np.ndarray:
        """Apply white balance correction."""
        if settings.wb_method == 'gray_world':
            return self._gray_world_wb(image)
        elif settings.wb_method == 'white_patch':
            return self._white_patch_wb(image)
        else:
            return self._manual_wb(image, settings.wb_temperature)

    def _gray_world_wb(self, image: np.ndarray) -> np.ndarray:
        """Apply gray world white balance."""
        result = image.astype(np.float32)

        # Calculate mean of each channel
        avg_b = np.mean(result[:, :, 2])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 0])

        # Calculate overall average
        avg = (avg_b + avg_g + avg_r) / 3

        # Scale each channel
        result[:, :, 0] = result[:, :, 0] * (avg / avg_r) if avg_r > 0 else result[:, :, 0]
        result[:, :, 1] = result[:, :, 1] * (avg / avg_g) if avg_g > 0 else result[:, :, 1]
        result[:, :, 2] = result[:, :, 2] * (avg / avg_b) if avg_b > 0 else result[:, :, 2]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _white_patch_wb(self, image: np.ndarray) -> np.ndarray:
        """Apply white patch white balance."""
        result = image.astype(np.float32)

        # Find maximum values (assumed to be white)
        max_r = np.percentile(result[:, :, 0], 99)
        max_g = np.percentile(result[:, :, 1], 99)
        max_b = np.percentile(result[:, :, 2], 99)

        # Scale to make white pixels (255, 255, 255)
        result[:, :, 0] = result[:, :, 0] * (255 / max_r) if max_r > 0 else result[:, :, 0]
        result[:, :, 1] = result[:, :, 1] * (255 / max_g) if max_g > 0 else result[:, :, 1]
        result[:, :, 2] = result[:, :, 2] * (255 / max_b) if max_b > 0 else result[:, :, 2]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _manual_wb(self, image: np.ndarray, temperature: float) -> np.ndarray:
        """Apply manual white balance adjustment."""
        result = image.astype(np.float32)

        # Temperature adjustment (warm/cool)
        # temperature > 1 = warmer (more red/yellow)
        # temperature < 1 = cooler (more blue)
        if temperature > 1:
            result[:, :, 0] *= temperature  # Increase red
            result[:, :, 2] /= temperature  # Decrease blue
        else:
            result[:, :, 0] /= (2 - temperature)  # Decrease red
            result[:, :, 2] *= (2 - temperature)  # Increase blue

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_exposure(self, image: np.ndarray,
                        settings: ProcessingSettings) -> np.ndarray:
        """Apply exposure and contrast adjustments."""
        result = image.astype(np.float32)

        # Apply exposure adjustment (simulates EV change)
        if settings.exposure_adjustment != 0:
            factor = 2 ** settings.exposure_adjustment
            result *= factor

        # Apply contrast
        if settings.contrast != 1.0:
            mean = 127.5
            result = (result - mean) * settings.contrast + mean

        # Apply brightness
        if settings.brightness != 0:
            result += settings.brightness

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_binarize(self, image: np.ndarray,
                        settings: ProcessingSettings) -> np.ndarray:
        """Apply binarization (convert to black and white)."""
        # Convert to grayscale first
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        if settings.binarize_method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif settings.binarize_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                settings.adaptive_block_size,
                settings.adaptive_c
            )
        elif settings.binarize_method == 'sauvola':
            binary = self._sauvola_threshold(gray)
        else:
            _, binary = cv2.threshold(gray, settings.binarize_threshold, 255, cv2.THRESH_BINARY)

        # Convert back to RGB
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    def _sauvola_threshold(self, gray: np.ndarray,
                           window_size: int = 25,
                           k: float = 0.5,
                           r: float = 128) -> np.ndarray:
        """Apply Sauvola binarization (better for documents)."""
        # Calculate local mean and standard deviation
        mean = cv2.blur(gray.astype(np.float32), (window_size, window_size))

        # Calculate squared image
        sq = gray.astype(np.float32) ** 2
        sq_mean = cv2.blur(sq, (window_size, window_size))

        # Standard deviation
        std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))

        # Sauvola threshold
        threshold = mean * (1 + k * (std / r - 1))

        # Apply threshold
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)

        return binary

    def get_processing_history(self) -> List[Tuple[str, np.ndarray]]:
        """Get the history of processing steps."""
        return self._processing_history.copy()

    def get_step_result(self, step_name: str) -> Optional[np.ndarray]:
        """Get the result of a specific processing step."""
        for name, image in self._processing_history:
            if name == step_name:
                return image.copy()
        return None

    def preview_step(self, image: np.ndarray,
                     step: ProcessingStep,
                     settings: Optional[ProcessingSettings] = None) -> np.ndarray:
        """
        Preview a single processing step without applying others.

        Args:
            image: Input image
            step: Processing step to preview
            settings: Optional settings

        Returns:
            Result of applying just this step
        """
        settings = settings or self.settings

        if step == ProcessingStep.DEWARP:
            return self._apply_dewarp(image, settings)
        elif step == ProcessingStep.PERSPECTIVE:
            return self._apply_perspective(image, settings)
        elif step == ProcessingStep.DESKEW:
            return self._apply_deskew(image, settings)
        elif step == ProcessingStep.CROP:
            return self._apply_crop(image, settings)
        elif step == ProcessingStep.WHITE_BALANCE:
            return self._apply_white_balance(image, settings)
        elif step == ProcessingStep.EXPOSURE:
            return self._apply_exposure(image, settings)
        elif step == ProcessingStep.BINARIZE:
            return self._apply_binarize(image, settings)

        return image.copy()

    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze an image and suggest processing settings.

        Returns:
            Dictionary with analysis results and recommendations
        """
        # Analyze curve
        curve_analysis = self.dewarper.analyze_curve_severity(image)

        # Detect perspective distortion
        corners = self.perspective_corrector.detect_page_corners(image)
        perspective_analysis = self.perspective_corrector.estimate_perspective_distortion(corners)

        # Detect skew
        skew_angle = self._detect_skew_angle(image)

        # Analyze exposure
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Build recommendations
        recommendations = []

        if curve_analysis.get('severity') in ['medium', 'high']:
            recommendations.append('Enable dewarping to correct page curvature')

        if perspective_analysis.get('distortion', 0) > 0.05:
            recommendations.append('Enable perspective correction to fix keystone distortion')

        if abs(skew_angle) > 0.5:
            recommendations.append(f'Enable deskew (detected angle: {skew_angle:.1f}°)')

        if mean_brightness < 100:
            recommendations.append('Image appears dark - consider exposure adjustment')
        elif mean_brightness > 200:
            recommendations.append('Image appears overexposed - consider reducing exposure')

        if std_brightness < 30:
            recommendations.append('Low contrast detected - consider contrast adjustment')

        return {
            'curve': curve_analysis,
            'perspective': perspective_analysis,
            'skew_angle': float(skew_angle),
            'brightness': {
                'mean': float(mean_brightness),
                'std': float(std_brightness)
            },
            'recommendations': recommendations,
            'detected_corners': corners.tolist() if corners is not None else None
        }
