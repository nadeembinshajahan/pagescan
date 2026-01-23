#!/usr/bin/env python3
"""
Test script for PageScan processing pipeline.
"""

import sys
import os
# Dynamic path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
from PIL import Image
import cv2

# Import core modules
from pagescan.core.image_loader import ImageLoader
from pagescan.core.dewarper import PageDewarper, DewarpParams
from pagescan.core.perspective import PerspectiveCorrector, PerspectiveParams
from pagescan.core.processor import ImageProcessor, ProcessingSettings


def test_image(image_path: str, output_prefix: str):
    """Test processing on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print('='*60)

    # Load image
    loader = ImageLoader()
    image = loader.load_image(image_path)

    if image is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return False

    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    # Initialize processor
    processor = ImageProcessor()

    # Analyze the image
    print("\n--- Image Analysis ---")
    analysis = processor.analyze_image(image)

    # Curve analysis
    curve = analysis.get('curve', {})
    if curve.get('detected'):
        print(f"Curve detected: {curve.get('severity')} severity, {curve.get('curve_direction')} direction")
        print(f"  Max deviation: {curve.get('max_deviation', 0):.1f} pixels")
    else:
        print("Curve: Not detected")

    # Perspective analysis
    perspective = analysis.get('perspective', {})
    print(f"Perspective distortion: {perspective.get('distortion', 0):.1%} ({perspective.get('type', 'unknown')})")
    if perspective.get('edge_lengths'):
        edges = perspective['edge_lengths']
        print(f"  Edge lengths - Top: {edges['top']:.0f}, Bottom: {edges['bottom']:.0f}")
        print(f"               Left: {edges['left']:.0f}, Right: {edges['right']:.0f}")

    # Skew
    skew = analysis.get('skew_angle', 0)
    print(f"Skew angle: {skew:.2f}°")

    # Brightness
    brightness = analysis.get('brightness', {})
    print(f"Brightness: mean={brightness.get('mean', 0):.1f}, std={brightness.get('std', 0):.1f}")

    # Recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")

    # Test individual processing steps
    print("\n--- Testing Dewarping ---")
    dewarper = PageDewarper()
    try:
        dewarped = dewarper.dewarp(image)
        print(f"Dewarping complete. Output size: {dewarped.shape[1]}x{dewarped.shape[0]}")

        # Save dewarped result
        output_prefix_name = os.path.basename(output_prefix)
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{output_prefix_name}_dewarped.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(dewarped, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Dewarping error: {e}")

    print("\n--- Testing Perspective Correction ---")
    perspective_corrector = PerspectiveCorrector()
    try:
        corners = perspective_corrector.detect_page_corners(image)
        if corners is not None:
            print(f"Detected corners:")
            labels = ['TL', 'TR', 'BR', 'BL']
            for label, corner in zip(labels, corners):
                print(f"  {label}: ({corner[0]:.0f}, {corner[1]:.0f})")

            corrected = perspective_corrector.correct(image, corners)
            print(f"Perspective correction complete. Output size: {corrected.shape[1]}x{corrected.shape[0]}")

            # Save perspective corrected result
            output_path = os.path.join(output_dir, f"{output_prefix_name}_perspective.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Saved: {output_path}")

            # Save corner preview
            preview = perspective_corrector.get_corner_preview(image, corners)
            preview_path = os.path.join(output_dir, f"{output_prefix_name}_corners_preview.jpg")
            cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Saved corner preview: {preview_path}")
        else:
            print("No page corners detected")
    except Exception as e:
        print(f"Perspective correction error: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing Full Pipeline ---")
    settings = ProcessingSettings()
    settings.enable_dewarp = True
    settings.enable_perspective = True
    settings.enable_deskew = True
    settings.enable_crop = True
    settings.enable_white_balance = False
    settings.enable_exposure = False
    settings.enable_binarize = False

    try:
        processed = processor.process(image, settings)
        print(f"Full processing complete. Output size: {processed.shape[1]}x{processed.shape[0]}")

        # Save fully processed result
        output_path = os.path.join(output_dir, f"{output_prefix_name}_processed.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {output_path}")

        # Get processing history
        history = processor.get_processing_history()
        print(f"Processing steps applied: {[step[0] for step in history]}")

    except Exception as e:
        print(f"Full pipeline error: {e}")
        import traceback
        traceback.print_exc()

    return True


def main():
    """Run tests on sample images."""
    print("PageScan Processing Pipeline Test")
    print("="*60)

    # Test with provided images
    test_images = [
        ("/tmp/test_book_1.jpg", "book1"),
        ("/tmp/test_book_2.jpg", "book2"),
    ]

    # Check which test images exist
    available_tests = []
    for path, prefix in test_images:
        if os.path.exists(path):
            available_tests.append((path, prefix))

    if not available_tests:
        print("No test images found. Looking for any images in test_images/")
        import glob
        test_images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images')
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.TIFF']:
            for path in glob.glob(os.path.join(test_images_dir, ext)):
                if '_processed' not in path and '_dewarped' not in path and '_perspective' not in path and '_corners_preview' not in path:
                    available_tests.append((path, os.path.splitext(os.path.basename(path))[0]))

    if not available_tests:
        print("No test images available.")
        print("Please provide images to test.")
        return

    print(f"Found {len(available_tests)} test image(s)")

    for image_path, prefix in available_tests:
        test_image(image_path, prefix)

    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == '__main__':
    main()
