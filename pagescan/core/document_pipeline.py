"""
Document processing pipeline.

Orchestrates the multi-step document processing:
1. Spine detection & split
2. Page polygon detection
3. Perspective correction
4. DL-based dewarping
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .spine_detector import SpineDetector, SpineDetectionResult
from .page_detector import PageDetector, PageBoundary
from .post_processor import PostProcessor


@dataclass
class ProcessedPage:
    """Result of processing a single page."""
    original: np.ndarray  # Original cropped page
    polygon: PageBoundary  # Detected polygon
    perspective_corrected: np.ndarray  # After perspective correction
    dewarped: Optional[np.ndarray] = None  # After DL dewarping
    post_processed: Optional[np.ndarray] = None  # After Booksorber-style processing
    page_number: int = 0  # 1 for left, 2 for right


@dataclass
class PipelineResult:
    """Result of the full document processing pipeline."""
    pages: List[ProcessedPage]
    spine_result: Optional[SpineDetectionResult] = None
    debug_images: dict = field(default_factory=dict)


class DocumentPipeline:
    """
    Multi-step document processing pipeline.
    
    Steps:
    1. Detect spine and split into pages
    2. Detect page boundary polygon for each page
    3. Apply perspective correction
    4. Apply DL dewarping (optional)
    """
    
    def __init__(self, use_dl_dewarp: bool = False):
        """
        Args:
            use_dl_dewarp: Whether to apply DL dewarping after perspective correction
        """
        self.spine_detector = SpineDetector()
        self.page_detector = PageDetector()
        self.post_processor = PostProcessor()
        self.use_dl_dewarp = use_dl_dewarp
        self._uvdoc = None
        
    def _get_uvdoc(self):
        """Lazy load UVDoc dewarper."""
        if self._uvdoc is None and self.use_dl_dewarp:
            try:
                from .uvdoc_dewarper import UVDocDewarper
                self._uvdoc = UVDocDewarper()
                if not self._uvdoc.is_available():
                    self._uvdoc = None
            except Exception as e:
                print(f"UVDoc not available: {e}")
                self._uvdoc = None
        return self._uvdoc
    
    def process(self, image: np.ndarray, 
                save_debug: bool = False,
                post_process_mode: Optional[str] = None) -> PipelineResult:
        """
        Process a document image through the full pipeline.
        """
        debug_images = {}
        h, w = image.shape[:2]
        
        # Step 1: Spine detection & split
        spine_result = self.spine_detector.detect_and_split(image)
        
        if save_debug:
            debug_images['spine_preview'] = self.spine_detector.get_preview(
                image, spine_result
            )
        
        # Process each page
        pages = []
        
        if spine_result.is_spread:
            # Create a full-size mask for the spine split
            # We want to strictly mask out the "other" side of the line
            # so the PageDetector doesn't see it (merged blobs).
            
            # Coordinate grid
            Y, X = np.indices((h, w))
            
            # Line equation x = x0 + t*vx, y = y0 + t*vy => derived from two points
            # Point 1: (spine_x_top, 0)
            # Point 2: (spine_x_bottom, h)
            # Line eq: (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1)
            # (x - x1)(y2 - y1) - (y - y1)(x2 - x1) = 0
            # Let's use the cross product form for "left of line" check
            
            x1, y1 = spine_result.spine_x_top, 0
            x2, y2 = spine_result.spine_x_bottom, h
            
            # Vector along line: (dx, dy) = (x2-x1, y2-y1)
            # Vector from p1 to point p: (x-x1, y-y1)
            # Cross product Z = dx*(y-y1) - dy*(x-x1)
            # If Z > 0, point is on one side, < 0 on the other.
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Z component of cross product (p_line x p_point)
            # This determines which side of the line a pixel is on
            side_mask = (dx * (Y - y1) - dy * (X - x1)) > 0
            
            # Determine which side is "left" (x should be smaller)
            # At Y=0 (top), smaller X is left.
            # Check a point clearly on the left: (0, 0)
            # val_0 = dx*(0-0) - dy*(0-x1) = -dy * -x1 = y2 * x1. Since y2=h > 0 and x1 > 0, val_0 > 0.
            # So > 0 corresponds to the Left side?
            # Let's verify. x1=100, x2=100 (vertical). dx=0, dy=h.
            # side_mask = 0 - h*(X-100) > 0 => -h*X + 100h > 0 => 100h > hX => 100 > X.
            # Yes, positive cross product means X is smaller (Left side).
            
            left_mask = side_mask
            right_mask = ~side_mask

            # For slanted spines, we crop "generously" to ensure we don't cut content
            # Left page: from 0 to max(spine_x_top, spine_x_bottom) + margin
            margin = int(w * 0.05) # Increased margin to be safe
            left_limit = min(w, max(spine_result.spine_x_top, spine_result.spine_x_bottom) + margin)
            
            # Apply Mask to Left Page Region
            left_page_full = image.copy()
            left_page_full[~left_mask] = 0 # Black out right side
            left_page_img = left_page_full[:, :left_limit].copy()
            
            # Right page: from min(spine_x_top, spine_x_bottom) - margin to end
            right_limit = max(0, min(spine_result.spine_x_top, spine_result.spine_x_bottom) - margin)
            
            # Apply Mask to Right Page Region
            right_page_full = image.copy()
            right_page_full[~right_mask] = 0 # Black out left side
            right_page_img = right_page_full[:, right_limit:].copy()
            
            page_configs = [
                (left_page_img, 1),
                (right_page_img, 2)
            ]
        else:
            page_configs = [(image, 1)]
        
        for page_img, page_num in page_configs:
            if page_img.shape[1] < 50:  # Skip tiny images
                continue
                
            processed = self._process_single_page(page_img, page_num, save_debug, post_process_mode)
            pages.append(processed)
            
            if save_debug:
                debug_images[f'page_{page_num}_polygon'] = self.page_detector.get_preview(
                    page_img, processed.polygon
                )
        
        return PipelineResult(
            pages=pages,
            spine_result=spine_result,
            debug_images=debug_images
        )
    
    def _process_single_page(self, image: np.ndarray, 
                             page_num: int,
                             save_debug: bool = False,
                             post_process_mode: Optional[str] = None) -> ProcessedPage:
        """Process a single page through polygon detection, perspective, and dewarp."""
        
        # Step 2: Detect page polygon
        boundary = self.page_detector.detect(image, page_num)
        
        # Step 3: Perspective correction
        perspective_corrected = self._apply_perspective_correction(
            image, boundary.corners
        )
        
        # Step 4: DL dewarping (optional)
        dewarped = None
        if self.use_dl_dewarp:
            uvdoc = self._get_uvdoc()
            if uvdoc is not None:
                try:
                    dewarped = uvdoc.dewarp(perspective_corrected)
                except Exception as e:
                    print(f"DL dewarp failed: {e}")
                    dewarped = perspective_corrected
            else:
                dewarped = perspective_corrected
        
        # Step 5: Post-processing (Booksorber-style)
        post_processed = None
        if post_process_mode:
            img_to_process = dewarped if dewarped is not None else perspective_corrected
            post_processed = self.post_processor.process(img_to_process, mode=post_process_mode)
        
        return ProcessedPage(
            original=image,
            polygon=boundary,
            perspective_corrected=perspective_corrected,
            dewarped=dewarped,
            post_processed=post_processed,
            page_number=page_num
        )
    
    def _apply_perspective_correction(self, image: np.ndarray,
                                      corners: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to get a rectangular output.
        
        Args:
            image: Input image
            corners: 4 corners as (TL, TR, BR, BL)
            
        Returns:
            Perspective-corrected rectangular image
        """
        # Calculate output dimensions
        # Width: max of top and bottom edge lengths
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        width = int(max(width_top, width_bottom))
        
        # Height: max of left and right edge lengths
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        height = int(max(height_left, height_right))
        
        # Ensure minimum size
        width = max(width, 100)
        height = max(height, 100)
        
        # Target corners (rectangle)
        dst_corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Get perspective transform
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
        
        # Apply transform
        result = cv2.warpPerspective(
            image, M, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return result
    
    def process_and_save(self, image: np.ndarray, 
                         output_dir: str, 
                         basename: str,
                         post_process_mode: Optional[str] = None) -> PipelineResult:
        """
        Process and save all outputs.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        result = self.process(image, save_debug=True, post_process_mode=post_process_mode)
        
        # Save debug images
        for name, img in result.debug_images.items():
            path = os.path.join(output_dir, f"{basename}_{name}.jpg")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Save processed pages
        for page in result.pages:
            prefix = f"{basename}_page_{page.page_number}"
            
            # Save final result (most processed)
            final = page.post_processed if page.post_processed is not None else \
                    (page.dewarped if page.dewarped is not None else page.perspective_corrected)
            
            # Convert to BGR if necessary
            if len(final.shape) == 3:
                final_bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
            else:
                final_bgr = final
                
            cv2.imwrite(os.path.join(output_dir, f"{prefix}_final.jpg"), final_bgr)
            
            # Save intermediate steps if they are distinct from final
            if page.post_processed is not None:
                # If we post-processed, save the steps before it
                if page.dewarped is not None:
                    cv2.imwrite(os.path.join(output_dir, f"{prefix}_dewarped.jpg"), 
                                cv2.cvtColor(page.dewarped, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(output_dir, f"{prefix}_perspective.jpg"), 
                            cv2.cvtColor(page.perspective_corrected, cv2.COLOR_RGB2BGR))
            elif page.dewarped is not None:
                # If dewarped is final, save perspective as intermediate
                cv2.imwrite(os.path.join(output_dir, f"{prefix}_perspective.jpg"), 
                            cv2.cvtColor(page.perspective_corrected, cv2.COLOR_RGB2BGR))
        
        return result
