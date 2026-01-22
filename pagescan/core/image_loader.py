"""
Image loading module with support for various formats including Canon CR3 RAW.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import cv2

# Try to import rawpy for RAW support
try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

# Try to import piexif for EXIF handling
try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False


SUPPORTED_FORMATS = {
    'standard': ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'],
    'raw': ['.cr3', '.cr2', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2']
}

ALL_SUPPORTED = SUPPORTED_FORMATS['standard'] + SUPPORTED_FORMATS['raw']


class ImageLoader:
    """Handles loading images from various formats including RAW."""

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}
        self._exif_cache: Dict[str, Dict] = {}
        self._max_cache_size = 50  # Maximum images to cache

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Return list of all supported file extensions."""
        return ALL_SUPPORTED.copy()

    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if a file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in ALL_SUPPORTED

    @staticmethod
    def is_raw(file_path: str) -> bool:
        """Check if a file is a RAW format."""
        ext = Path(file_path).suffix.lower()
        return ext in SUPPORTED_FORMATS['raw']

    def load_image(self, file_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Load an image from file.

        Args:
            file_path: Path to the image file
            use_cache: Whether to use caching

        Returns:
            Image as numpy array in RGB format, or None if loading fails
        """
        file_path = str(Path(file_path).resolve())

        # Check cache
        if use_cache and file_path in self._cache:
            return self._cache[file_path].copy()

        if not os.path.exists(file_path):
            return None

        ext = Path(file_path).suffix.lower()

        try:
            if ext in SUPPORTED_FORMATS['raw']:
                image = self._load_raw(file_path)
            else:
                image = self._load_standard(file_path)

            if image is not None and use_cache:
                self._manage_cache()
                self._cache[file_path] = image.copy()

            return image

        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None

    def _load_standard(self, file_path: str) -> Optional[np.ndarray]:
        """Load a standard image format (JPEG, PNG, TIFF, etc.)."""
        # Use PIL for better format support and EXIF handling
        try:
            with Image.open(file_path) as img:
                # Handle EXIF orientation
                img = self._apply_exif_orientation(img)

                # Convert to RGB if necessary
                if img.mode == 'RGBA':
                    # Create white background for transparency
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Convert to numpy array
                return np.array(img)

        except Exception as e:
            # Fallback to OpenCV
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None

    def _load_raw(self, file_path: str) -> Optional[np.ndarray]:
        """Load a RAW image file."""
        if not HAS_RAWPY:
            print("rawpy not installed. Cannot load RAW files.")
            return None

        try:
            with rawpy.imread(file_path) as raw:
                # Post-process with good defaults for book scanning
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=False,
                    output_bps=8,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
                )
                return rgb

        except Exception as e:
            print(f"Error loading RAW file {file_path}: {e}")
            return None

    def _apply_exif_orientation(self, img: Image.Image) -> Image.Image:
        """Apply EXIF orientation to image."""
        try:
            exif = img.getexif()
            if exif:
                orientation = exif.get(274)  # 274 is the orientation tag
                if orientation:
                    rotations = {
                        3: Image.Transpose.ROTATE_180,
                        6: Image.Transpose.ROTATE_270,
                        8: Image.Transpose.ROTATE_90,
                    }
                    if orientation in rotations:
                        img = img.transpose(rotations[orientation])
                    elif orientation == 2:
                        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    elif orientation == 4:
                        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    elif orientation == 5:
                        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        img = img.transpose(Image.Transpose.ROTATE_270)
                    elif orientation == 7:
                        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        img = img.transpose(Image.Transpose.ROTATE_90)
        except Exception:
            pass
        return img

    def load_thumbnail(self, file_path: str, max_size: int = 200) -> Optional[np.ndarray]:
        """
        Load a thumbnail of the image.

        Args:
            file_path: Path to the image file
            max_size: Maximum dimension of thumbnail

        Returns:
            Thumbnail as numpy array in RGB format
        """
        ext = Path(file_path).suffix.lower()

        try:
            # For RAW files, load full image and resize
            if ext in SUPPORTED_FORMATS['raw']:
                full_image = self.load_image(file_path, use_cache=False)
                if full_image is not None:
                    return self._resize_to_thumbnail(full_image, max_size)
                return None

            # For standard formats, use PIL's thumbnail feature
            with Image.open(file_path) as img:
                img = self._apply_exif_orientation(img)
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)

        except Exception as e:
            print(f"Error loading thumbnail for {file_path}: {e}")
            return None

    def _resize_to_thumbnail(self, image: np.ndarray, max_size: int) -> np.ndarray:
        """Resize image to thumbnail size."""
        h, w = image.shape[:2]
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def get_exif(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract EXIF metadata from image.

        Returns:
            Dictionary of EXIF data or None
        """
        file_path = str(Path(file_path).resolve())

        if file_path in self._exif_cache:
            return self._exif_cache[file_path].copy()

        if not HAS_PIEXIF:
            return None

        try:
            exif_dict = piexif.load(file_path)

            # Convert to more usable format
            result = {}
            for ifd_name in exif_dict:
                if isinstance(exif_dict[ifd_name], dict):
                    for tag, value in exif_dict[ifd_name].items():
                        tag_name = piexif.TAGS.get(ifd_name, {}).get(tag, {}).get('name', str(tag))
                        result[tag_name] = value

            self._exif_cache[file_path] = result
            return result.copy()

        except Exception:
            return None

    def get_image_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get basic image information without loading full image.

        Returns:
            Dictionary with image dimensions, format, etc.
        """
        try:
            ext = Path(file_path).suffix.lower()

            if ext in SUPPORTED_FORMATS['raw']:
                if not HAS_RAWPY:
                    return None
                with rawpy.imread(file_path) as raw:
                    return {
                        'width': raw.sizes.width,
                        'height': raw.sizes.height,
                        'format': ext[1:].upper(),
                        'is_raw': True
                    }
            else:
                with Image.open(file_path) as img:
                    return {
                        'width': img.width,
                        'height': img.height,
                        'format': img.format,
                        'mode': img.mode,
                        'is_raw': False
                    }

        except Exception:
            return None

    def _manage_cache(self):
        """Remove oldest items if cache is full."""
        if len(self._cache) >= self._max_cache_size:
            # Remove first (oldest) item
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    def clear_cache(self):
        """Clear the image cache."""
        self._cache.clear()
        self._exif_cache.clear()

    def scan_folder(self, folder_path: str, recursive: bool = False) -> List[str]:
        """
        Scan a folder for supported image files.

        Args:
            folder_path: Path to the folder
            recursive: Whether to scan subfolders

        Returns:
            List of file paths sorted by name
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            return []

        files = []
        pattern = '**/*' if recursive else '*'

        for ext in ALL_SUPPORTED:
            files.extend(folder.glob(f'{pattern}{ext}'))
            files.extend(folder.glob(f'{pattern}{ext.upper()}'))

        # Remove duplicates and sort
        files = sorted(set(str(f) for f in files))
        return files
