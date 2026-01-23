import cv2
import numpy as np
from typing import Optional, Tuple

class PostProcessor:
    """
    Advanced post-processing for book scans, matching Booksorber's strengths.
    Includes shadow removal, adaptive binarization, and micro-deskewing.
    """
    
    def __init__(self):
        pass

    def remove_gutter_shadow(self, image: np.ndarray) -> np.ndarray:
        """
        Removes dark shadows in the page fold by normalizing illumination.
        Works by estimating the background and dividing the image by it.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Estimate background (illumination map)
        # Use a large-scale closing or dilation to get the paper color
        kernel_size = max(51, min(image.shape[0], image.shape[1]) // 20)
        if kernel_size % 2 == 0: kernel_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        background = cv2.dilate(gray, kernel)
        background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)
        
        # Avoid division by zero
        background = np.maximum(background, 1)
        
        # Divide original by background and scale back to 0-255
        normalized = np.clip((gray.astype(np.float32) / background.astype(np.float32)) * 255, 0, 255).astype(np.uint8)
        
        # If input was color, apply the gain to all channels
        if len(image.shape) == 3:
            gain = normalized.astype(np.float32) / np.maximum(gray.astype(np.float32), 1)
            result = np.clip(image.astype(np.float32) * gain[:, :, np.newaxis], 0, 255).astype(np.uint8)
            return result
        
        return normalized

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Produces a crisp black and white scan using adaptive thresholding.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Adaptive thresholding (Sauvola-like behavior with cv2.adaptiveThreshold)
        # Block size should be large enough to cover multiple text lines
        block_size = 41
        c = 15
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c
        )
        return binary

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Performs micro-deskewing by aligning text lines horizontally.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Find edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use Hough lines to find dominant orientation
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return image
            
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            # Only consider near-horizontal lines
            if abs(angle) < 15:
                angles.append(angle)
                
        if not angles:
            return image
            
        median_angle = np.median(angles)
        
        if abs(median_angle) < 0.1:
            return image
            
        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated

    def process(self, image: np.ndarray, mode: str = 'clean') -> np.ndarray:
        """
        Main processing entry point.
        Modes: 'clean' (color + shadow removal), 'scan' (binarized), 'raw' (as-is)
        """
        result = image.copy()
        
        # 1. Deskew (Always good for professional look)
        result = self.deskew(result)
        
        # 2. Shadow removal
        if mode in ['clean', 'scan']:
            result = self.remove_gutter_shadow(result)
            
        # 3. Binarization
        if mode == 'scan':
            result = self.binarize(result)
            
        return result
