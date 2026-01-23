#!/usr/bin/env python3
"""
Test script for the multi-step document processing pipeline.

Tests the complete flow:
1. Spine detection & split
2. Page polygon detection  
3. Perspective correction
4. DL dewarping (if available)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from pagescan.core.document_pipeline import DocumentPipeline


def test_pipeline(image_path: str, output_dir: str):
    """Test the pipeline on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print('='*60)
    
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"  ERROR: Could not load image")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"  Image size: {img_rgb.shape[1]}x{img_rgb.shape[0]}")
    
    # Create pipeline
    pipeline = DocumentPipeline(use_dl_dewarp=True)
    
    # Process
    basename = os.path.splitext(os.path.basename(image_path))[0]
    result = pipeline.process_and_save(img_rgb, output_dir, basename)
    
    # Print results
    if result.spine_result:
        sr = result.spine_result
        print(f"  Spine detected: {sr.is_spread}")
        if sr.is_spread:
            print(f"    Spine position: {sr.spine_x}")
            print(f"    Confidence: {sr.confidence:.2f}")
    
    print(f"  Pages processed: {len(result.pages)}")
    for page in result.pages:
        print(f"    Page {page.page_number}:")
        print(f"      Original: {page.original.shape[1]}x{page.original.shape[0]}")
        print(f"      Polygon corners: {page.polygon.corners.tolist()}")
        print(f"      Perspective corrected: {page.perspective_corrected.shape[1]}x{page.perspective_corrected.shape[0]}")
        if page.dewarped is not None:
            print(f"      Dewarped: {page.dewarped.shape[1]}x{page.dewarped.shape[0]}")
        else:
            print(f"      Dewarped: N/A (DL not available)")
    
    print(f"  Debug images saved: {list(result.debug_images.keys())}")
    print(f"  Output directory: {output_dir}")
    

def main():
    # Setup paths
    test_dir = Path(__file__).parent / "test_images"
    output_dir = test_dir / "pipeline_results"
    output_dir.mkdir(exist_ok=True)
    
    # Find test images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    test_images = [
        f for f in test_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not test_images:
        print("No test images found in test_images/")
        return
    
    print(f"Found {len(test_images)} test images")
    print(f"Output directory: {output_dir}")
    
    # Test each image
    for img_path in sorted(test_images):
        try:
            test_pipeline(str(img_path), str(output_dir))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Pipeline testing complete!")
    print(f"Results saved to: {output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
