"""
Image export module with support for various formats and batch processing.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image
import cv2

try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False


class ExportFormat(Enum):
    """Supported export formats."""
    JPEG = 'jpeg'
    PNG = 'png'
    TIFF = 'tiff'


@dataclass
class ExportSettings:
    """Settings for image export."""
    # Output format
    format: ExportFormat = ExportFormat.JPEG

    # Output directory
    output_dir: str = ''

    # JPEG quality (1-100)
    jpeg_quality: int = 95

    # PNG compression (0-9, 9 = max compression)
    png_compression: int = 6

    # TIFF compression
    tiff_compression: str = 'lzw'  # 'none', 'lzw', 'deflate'

    # Naming convention
    prefix: str = 'page_'
    start_number: int = 1
    zero_padding: int = 3  # e.g., 001, 002

    # EXIF handling
    preserve_exif: bool = True
    add_processing_info: bool = False

    # Resize options
    resize_enabled: bool = False
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    resize_method: str = 'lanczos'  # 'lanczos', 'bilinear', 'nearest'


class ImageExporter:
    """Handles exporting processed images."""

    def __init__(self, settings: Optional[ExportSettings] = None):
        self.settings = settings or ExportSettings()
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """
        Set callback for progress updates.

        Callback signature: (current: int, total: int, filename: str)
        """
        self._progress_callback = callback

    def export_single(self, image: np.ndarray,
                      filename: str,
                      settings: Optional[ExportSettings] = None,
                      exif_data: Optional[Dict] = None) -> str:
        """
        Export a single image.

        Args:
            image: Image as RGB numpy array
            filename: Output filename (without extension)
            settings: Optional settings override
            exif_data: Optional EXIF data to preserve

        Returns:
            Full path to exported file
        """
        settings = settings or self.settings

        # Apply resize if enabled
        if settings.resize_enabled:
            image = self._resize_image(image, settings)

        # Determine output path
        ext = self._get_extension(settings.format)
        output_path = os.path.join(settings.output_dir, f"{filename}{ext}")

        # Ensure output directory exists
        os.makedirs(settings.output_dir, exist_ok=True)

        # Export based on format
        if settings.format == ExportFormat.JPEG:
            self._export_jpeg(image, output_path, settings, exif_data)
        elif settings.format == ExportFormat.PNG:
            self._export_png(image, output_path, settings)
        elif settings.format == ExportFormat.TIFF:
            self._export_tiff(image, output_path, settings)

        return output_path

    def export_batch(self, images: List[np.ndarray],
                     settings: Optional[ExportSettings] = None,
                     exif_data_list: Optional[List[Dict]] = None) -> List[str]:
        """
        Export multiple images with sequential naming.

        Args:
            images: List of images as RGB numpy arrays
            settings: Optional settings override
            exif_data_list: Optional list of EXIF data for each image

        Returns:
            List of exported file paths
        """
        settings = settings or self.settings
        exported_paths = []

        for i, image in enumerate(images):
            # Generate filename
            number = settings.start_number + i
            filename = f"{settings.prefix}{number:0{settings.zero_padding}d}"

            # Get EXIF if available
            exif_data = None
            if exif_data_list and i < len(exif_data_list):
                exif_data = exif_data_list[i]

            # Export
            path = self.export_single(image, filename, settings, exif_data)
            exported_paths.append(path)

            # Progress callback
            if self._progress_callback:
                self._progress_callback(i + 1, len(images), filename)

        return exported_paths

    def _resize_image(self, image: np.ndarray,
                      settings: ExportSettings) -> np.ndarray:
        """Resize image if needed."""
        h, w = image.shape[:2]

        # Calculate new dimensions
        new_w, new_h = w, h

        if settings.max_width and w > settings.max_width:
            scale = settings.max_width / w
            new_w = settings.max_width
            new_h = int(h * scale)

        if settings.max_height and new_h > settings.max_height:
            scale = settings.max_height / new_h
            new_h = settings.max_height
            new_w = int(new_w * scale)

        if new_w == w and new_h == h:
            return image

        # Choose interpolation method
        interpolation_methods = {
            'lanczos': cv2.INTER_LANCZOS4,
            'bilinear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST
        }
        interpolation = interpolation_methods.get(settings.resize_method, cv2.INTER_LANCZOS4)

        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    def _get_extension(self, format: ExportFormat) -> str:
        """Get file extension for format."""
        extensions = {
            ExportFormat.JPEG: '.jpg',
            ExportFormat.PNG: '.png',
            ExportFormat.TIFF: '.tiff'
        }
        return extensions.get(format, '.jpg')

    def _export_jpeg(self, image: np.ndarray,
                     output_path: str,
                     settings: ExportSettings,
                     exif_data: Optional[Dict] = None):
        """Export as JPEG."""
        # Convert RGB to PIL Image
        pil_image = Image.fromarray(image)

        # Prepare EXIF data
        exif_bytes = None
        if settings.preserve_exif and exif_data and HAS_PIEXIF:
            try:
                # Build EXIF bytes
                exif_dict = {'0th': {}, 'Exif': {}, 'GPS': {}, '1st': {}}

                # Copy relevant EXIF data
                for key, value in exif_data.items():
                    if isinstance(value, bytes):
                        continue  # Skip binary data that might cause issues
                    # Add to appropriate IFD
                    # This is simplified; full implementation would map tags properly

                exif_bytes = piexif.dump(exif_dict)
            except Exception:
                exif_bytes = None

        # Save
        save_kwargs = {
            'quality': settings.jpeg_quality,
            'optimize': True
        }
        if exif_bytes:
            save_kwargs['exif'] = exif_bytes

        pil_image.save(output_path, 'JPEG', **save_kwargs)

    def _export_png(self, image: np.ndarray,
                    output_path: str,
                    settings: ExportSettings):
        """Export as PNG."""
        pil_image = Image.fromarray(image)
        pil_image.save(output_path, 'PNG', compress_level=settings.png_compression)

    def _export_tiff(self, image: np.ndarray,
                     output_path: str,
                     settings: ExportSettings):
        """Export as TIFF."""
        pil_image = Image.fromarray(image)

        compression_map = {
            'none': None,
            'lzw': 'tiff_lzw',
            'deflate': 'tiff_deflate'
        }
        compression = compression_map.get(settings.tiff_compression)

        save_kwargs = {}
        if compression:
            save_kwargs['compression'] = compression

        pil_image.save(output_path, 'TIFF', **save_kwargs)

    def get_estimated_file_size(self, image: np.ndarray,
                                settings: Optional[ExportSettings] = None) -> int:
        """
        Estimate the output file size in bytes.

        This is a rough estimate based on format and quality.
        """
        settings = settings or self.settings
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        raw_size = h * w * channels

        if settings.format == ExportFormat.JPEG:
            # JPEG compression ratio varies, estimate based on quality
            compression_ratio = 0.05 + (settings.jpeg_quality / 100) * 0.15
            return int(raw_size * compression_ratio)
        elif settings.format == ExportFormat.PNG:
            # PNG is lossless but compressed
            compression_ratio = 0.3 + (settings.png_compression / 9) * 0.2
            return int(raw_size * compression_ratio)
        elif settings.format == ExportFormat.TIFF:
            if settings.tiff_compression == 'none':
                return raw_size
            else:
                return int(raw_size * 0.5)

        return raw_size


class BatchProcessor:
    """Handles batch processing of multiple images."""

    def __init__(self):
        self._progress_callback: Optional[Callable[[int, int, str, str], None]] = None
        self._cancel_requested = False

    def set_progress_callback(self, callback: Callable[[int, int, str, str], None]):
        """
        Set callback for progress updates.

        Callback signature: (current: int, total: int, filename: str, status: str)
        """
        self._progress_callback = callback

    def request_cancel(self):
        """Request cancellation of batch processing."""
        self._cancel_requested = True

    def process_batch(self, file_paths: List[str],
                      processor: 'ImageProcessor',
                      exporter: ImageExporter,
                      loader: 'ImageLoader') -> Dict[str, Any]:
        """
        Process and export multiple images.

        Args:
            file_paths: List of input file paths
            processor: ImageProcessor instance
            exporter: ImageExporter instance
            loader: ImageLoader instance

        Returns:
            Dictionary with processing results and statistics
        """
        from .image_loader import ImageLoader
        from .processor import ImageProcessor

        self._cancel_requested = False
        results = {
            'total': len(file_paths),
            'processed': 0,
            'failed': 0,
            'cancelled': False,
            'output_files': [],
            'errors': []
        }

        for i, file_path in enumerate(file_paths):
            if self._cancel_requested:
                results['cancelled'] = True
                break

            filename = Path(file_path).stem

            if self._progress_callback:
                self._progress_callback(i + 1, len(file_paths), filename, 'loading')

            try:
                # Load image
                image = loader.load_image(file_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {file_path}")

                # Get EXIF data
                exif_data = loader.get_exif(file_path)

                if self._progress_callback:
                    self._progress_callback(i + 1, len(file_paths), filename, 'processing')

                # Process image
                processed = processor.process(image)

                if self._progress_callback:
                    self._progress_callback(i + 1, len(file_paths), filename, 'exporting')

                # Export
                output_filename = f"{exporter.settings.prefix}{i + exporter.settings.start_number:0{exporter.settings.zero_padding}d}"
                output_path = exporter.export_single(processed, output_filename, exif_data=exif_data)

                results['output_files'].append(output_path)
                results['processed'] += 1

                if self._progress_callback:
                    self._progress_callback(i + 1, len(file_paths), filename, 'done')

            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'file': file_path,
                    'error': str(e)
                })

                if self._progress_callback:
                    self._progress_callback(i + 1, len(file_paths), filename, f'error: {e}')

        return results
