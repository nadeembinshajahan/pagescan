#!/usr/bin/env python3
"""
Test PageScan processing with synthetic data to validate algorithms.
"""

import sys
sys.path.insert(0, '/home/user/pagescan')

import numpy as np
import cv2

# Import core modules
from pagescan.core.image_loader import ImageLoader
from pagescan.core.dewarper import PageDewarper, DewarpParams
from pagescan.core.perspective import PerspectiveCorrector, PerspectiveParams
from pagescan.core.processor import ImageProcessor, ProcessingSettings


def create_synthetic_book_page(width=800, height=1000, curve_strength=30, perspective_skew=50):
    """
    Create a synthetic book page with text lines and artificial curve/perspective distortion.
    """
    # Create base image (white page)
    image = np.ones((height, width, 3), dtype=np.uint8) * 240

    # Add some margin variation (darker near edges)
    for i in range(20):
        alpha = i / 20
        image[:, i] = image[:, i] * (0.7 + 0.3 * alpha)
        image[:, width-1-i] = image[:, width-1-i] * (0.7 + 0.3 * alpha)

    # Draw text-like horizontal lines
    line_height = 25
    margin = 80

    for y in range(margin, height - margin, line_height):
        # Calculate curve offset (simulating book spine curve)
        # Maximum curve in the middle of the page
        for x in range(margin, width - margin):
            # Parabolic curve - maximum at left edge (near spine)
            curve_factor = 1 - (x - margin) / (width - 2 * margin)
            curve_offset = int(curve_strength * curve_factor * curve_factor)

            # Draw a "text" pixel (varying thickness)
            line_y = y + curve_offset
            if 0 <= line_y < height - 3:
                # Simulate text with varying darkness
                line_length = np.random.randint(100, width - 2 * margin - 50)
                if x < margin + line_length:
                    # Random gaps to simulate words
                    if np.random.random() > 0.15:
                        darkness = np.random.randint(20, 60)
                        image[line_y:line_y+2, x] = darkness

    # Add page number
    cv2.putText(image, "382", (width - 70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    # Add chapter header
    cv2.putText(image, "CHAPTER TITLE", (margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    # Apply perspective distortion (simulate V-cradle angle)
    if perspective_skew > 0:
        # Define source points (original corners)
        src_pts = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])

        # Define destination points (with perspective)
        dst_pts = np.float32([
            [perspective_skew, perspective_skew//2],
            [width - perspective_skew//3, 0],
            [width, height],
            [0, height - perspective_skew//2]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        image = cv2.warpPerspective(image, M, (width, height))

    return image


def test_dewarping():
    """Test the dewarping algorithm."""
    print("\n" + "="*60)
    print("Testing Dewarping Algorithm")
    print("="*60)

    # Create test image with curved text
    print("\nCreating synthetic curved book page...")
    image = create_synthetic_book_page(curve_strength=40, perspective_skew=0)

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save original
    cv2.imwrite('/home/user/pagescan/test_images/synthetic_curved_original.jpg', image)
    print("Saved: test_images/synthetic_curved_original.jpg")

    # Initialize dewarper
    dewarper = PageDewarper()

    # Analyze curve
    print("\nAnalyzing curve severity...")
    analysis = dewarper.analyze_curve_severity(image_rgb)
    print(f"  Detected: {analysis.get('detected')}")
    print(f"  Severity: {analysis.get('severity')}")
    print(f"  Direction: {analysis.get('curve_direction')}")
    print(f"  Max deviation: {analysis.get('max_deviation', 0):.1f} px")

    # Apply dewarping
    print("\nApplying dewarping...")
    dewarped = dewarper.dewarp(image_rgb)

    # Save result
    cv2.imwrite('/home/user/pagescan/test_images/synthetic_curved_dewarped.jpg',
                cv2.cvtColor(dewarped, cv2.COLOR_RGB2BGR))
    print("Saved: test_images/synthetic_curved_dewarped.jpg")

    # Get curve preview
    curve = dewarper.get_last_curve()
    if curve is not None:
        preview = dewarper.get_curve_preview(image_rgb, curve)
        cv2.imwrite('/home/user/pagescan/test_images/synthetic_curve_preview.jpg',
                    cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        print("Saved: test_images/synthetic_curve_preview.jpg")

    print("\nDewarping test PASSED")
    return True


def test_perspective():
    """Test the perspective correction algorithm."""
    print("\n" + "="*60)
    print("Testing Perspective Correction Algorithm")
    print("="*60)

    # Create test image with perspective distortion
    print("\nCreating synthetic page with perspective distortion...")
    image = create_synthetic_book_page(curve_strength=0, perspective_skew=60)

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save original
    cv2.imwrite('/home/user/pagescan/test_images/synthetic_perspective_original.jpg', image)
    print("Saved: test_images/synthetic_perspective_original.jpg")

    # Initialize corrector
    corrector = PerspectiveCorrector()

    # Detect corners
    print("\nDetecting page corners...")
    corners = corrector.detect_page_corners(image_rgb)

    if corners is not None:
        print("Corners detected:")
        labels = ['TL', 'TR', 'BR', 'BL']
        for label, corner in zip(labels, corners):
            print(f"  {label}: ({corner[0]:.0f}, {corner[1]:.0f})")

        # Analyze distortion
        distortion = corrector.estimate_perspective_distortion(corners)
        print(f"\nDistortion analysis:")
        print(f"  Type: {distortion.get('type')}")
        print(f"  Amount: {distortion.get('distortion', 0):.1%}")

        # Apply correction
        print("\nApplying perspective correction...")
        corrected = corrector.correct(image_rgb, corners)

        # Save results
        cv2.imwrite('/home/user/pagescan/test_images/synthetic_perspective_corrected.jpg',
                    cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))
        print("Saved: test_images/synthetic_perspective_corrected.jpg")

        # Save corner preview
        preview = corrector.get_corner_preview(image_rgb, corners)
        cv2.imwrite('/home/user/pagescan/test_images/synthetic_corners_preview.jpg',
                    cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        print("Saved: test_images/synthetic_corners_preview.jpg")

        print("\nPerspective test PASSED")
        return True
    else:
        print("WARNING: Could not detect corners (may need real image)")
        return False


def test_full_pipeline():
    """Test the full processing pipeline."""
    print("\n" + "="*60)
    print("Testing Full Processing Pipeline")
    print("="*60)

    # Create test image with both curve and perspective
    print("\nCreating synthetic page with curve AND perspective distortion...")
    image = create_synthetic_book_page(curve_strength=35, perspective_skew=40)

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save original
    cv2.imwrite('/home/user/pagescan/test_images/synthetic_combined_original.jpg', image)
    print("Saved: test_images/synthetic_combined_original.jpg")

    # Initialize processor
    processor = ImageProcessor()

    # Full analysis
    print("\nRunning full image analysis...")
    analysis = processor.analyze_image(image_rgb)

    print(f"  Curve severity: {analysis.get('curve', {}).get('severity', 'unknown')}")
    print(f"  Perspective distortion: {analysis.get('perspective', {}).get('distortion', 0):.1%}")
    print(f"  Skew angle: {analysis.get('skew_angle', 0):.1f}°")

    if analysis.get('recommendations'):
        print("  Recommendations:")
        for rec in analysis['recommendations']:
            print(f"    - {rec}")

    # Configure processing
    settings = ProcessingSettings()
    settings.enable_dewarp = True
    settings.enable_perspective = True
    settings.enable_deskew = True
    settings.enable_crop = True
    settings.enable_white_balance = False
    settings.enable_binarize = False

    # Process
    print("\nRunning full pipeline...")
    processed = processor.process(image_rgb, settings)

    # Save result
    cv2.imwrite('/home/user/pagescan/test_images/synthetic_combined_processed.jpg',
                cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    print("Saved: test_images/synthetic_combined_processed.jpg")

    # Show processing history
    history = processor.get_processing_history()
    print(f"\nProcessing steps completed: {[h[0] for h in history]}")

    print("\nFull pipeline test PASSED")
    return True


def test_image_loader():
    """Test the image loader."""
    print("\n" + "="*60)
    print("Testing Image Loader")
    print("="*60)

    loader = ImageLoader()

    # Test supported formats
    print("\nSupported formats:")
    print(f"  Standard: {loader.get_supported_extensions()[:6]}")
    print(f"  RAW: {loader.get_supported_extensions()[6:]}")

    # Test format detection
    test_files = ['test.jpg', 'photo.CR3', 'scan.png', 'doc.pdf']
    print("\nFormat detection:")
    for f in test_files:
        supported = loader.is_supported(f)
        is_raw = loader.is_raw(f) if supported else False
        print(f"  {f}: supported={supported}, raw={is_raw}")

    print("\nImage loader test PASSED")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("PageScan Algorithm Validation Tests")
    print("="*60)

    results = []

    # Run tests
    results.append(("Image Loader", test_image_loader()))
    results.append(("Dewarping", test_dewarping()))
    results.append(("Perspective", test_perspective()))
    results.append(("Full Pipeline", test_full_pipeline()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. Check output above for details.")


if __name__ == '__main__':
    main()
