#!/usr/bin/env python3
"""
Test script for patch-based color detection.

Creates synthetic images with known colors and verifies that the
patch-based histogram voting method correctly identifies them.
"""

import cv2
import numpy as np
import sys


def create_colored_box(color_bgr, size=(100, 100)):
    """Create a solid colored image."""
    return np.full((size[0], size[1], 3), color_bgr, dtype=np.uint8)


def test_color_detection():
    """Test the patch-based color detection with synthetic images."""

    # Create a dummy filter instance (we only need the color detection methods)
    class DummyFilter:
        def __init__(self):
            pass

        def _get_patchwise_dominant_color(self, crop_rgb, grid_size=4):
            # Copy the method from ReferringDetectionFilter
            hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
            h, w = hsv.shape[:2]

            if h < grid_size or w < grid_size:
                grid_size = min(h, w, 2)

            patch_h = max(1, h // grid_size)
            patch_w = max(1, w // grid_size)
            color_votes = {}

            def classify_pixel_color(hsv_pixel):
                h_val, s_val, v_val = hsv_pixel

                if v_val < 50:
                    return 'black'
                elif v_val > 200 and s_val < 30:
                    return 'white'
                elif s_val < 25:
                    return 'gray'
                elif s_val >= 25:
                    if h_val < 15 or h_val >= 160:
                        return 'red'
                    elif 15 <= h_val < 25:
                        return 'orange'
                    elif 25 <= h_val < 35:
                        return 'yellow'
                    elif 35 <= h_val < 85:
                        return 'green'
                    elif 85 <= h_val < 160:
                        return 'blue'

                return 'unknown'

            for i in range(grid_size):
                for j in range(grid_size):
                    y_start = i * patch_h
                    y_end = min((i + 1) * patch_h, h)
                    x_start = j * patch_w
                    x_end = min((j + 1) * patch_w, w)

                    patch = hsv[y_start:y_end, x_start:x_end]

                    if patch.size > 0:
                        mean_pixel = np.mean(patch.reshape(-1, 3), axis=0)
                        color = classify_pixel_color(mean_pixel)
                        color_votes[color] = color_votes.get(color, 0) + 1

            if not color_votes:
                return 'unknown'

            return max(color_votes.items(), key=lambda x: x[1])[0]

    filter_obj = DummyFilter()

    # Test colors (BGR format)
    test_cases = [
        ((0, 0, 0), 'black', 'Black'),
        ((255, 255, 255), 'white', 'White'),
        ((128, 128, 128), 'gray', 'Gray'),
        ((0, 0, 255), 'red', 'Red'),
        ((0, 128, 255), 'orange', 'Orange'),
        ((0, 255, 255), 'yellow', 'Yellow'),
        ((0, 255, 0), 'green', 'Green'),
        ((255, 0, 0), 'blue', 'Blue'),
    ]

    print("Testing Patch-Based Color Detection")
    print("=" * 50)

    passed = 0
    failed = 0

    for bgr_color, expected_color, name in test_cases:
        # Create image in BGR
        img_bgr = create_colored_box(bgr_color)
        # Convert to RGB for processing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Detect color
        detected = filter_obj._get_patchwise_dominant_color(img_rgb)

        # Check if correct
        is_correct = detected == expected_color
        status = "✓ PASS" if is_correct else "✗ FAIL"

        if is_correct:
            passed += 1
        else:
            failed += 1

        print(f"{status} | {name:10s} | Expected: {expected_color:8s} | Detected: {detected:8s}")

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")

    return failed == 0


def test_mixed_colors():
    """Test detection on images with multiple colors (simulating real objects)."""
    print("\n\nTesting Mixed Color Scenarios")
    print("=" * 50)

    class DummyFilter:
        def __init__(self):
            pass

        def _get_patchwise_dominant_color(self, crop_rgb, grid_size=4):
            hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
            h, w = hsv.shape[:2]

            if h < grid_size or w < grid_size:
                grid_size = min(h, w, 2)

            patch_h = max(1, h // grid_size)
            patch_w = max(1, w // grid_size)
            color_votes = {}

            def classify_pixel_color(hsv_pixel):
                h_val, s_val, v_val = hsv_pixel

                if v_val < 50:
                    return 'black'
                elif v_val > 200 and s_val < 30:
                    return 'white'
                elif s_val < 25:
                    return 'gray'
                elif s_val >= 25:
                    if h_val < 15 or h_val >= 160:
                        return 'red'
                    elif 15 <= h_val < 25:
                        return 'orange'
                    elif 25 <= h_val < 35:
                        return 'yellow'
                    elif 35 <= h_val < 85:
                        return 'green'
                    elif 85 <= h_val < 160:
                        return 'blue'

                return 'unknown'

            for i in range(grid_size):
                for j in range(grid_size):
                    y_start = i * patch_h
                    y_end = min((i + 1) * patch_h, h)
                    x_start = j * patch_w
                    x_end = min((j + 1) * patch_w, w)

                    patch = hsv[y_start:y_end, x_start:x_end]

                    if patch.size > 0:
                        mean_pixel = np.mean(patch.reshape(-1, 3), axis=0)
                        color = classify_pixel_color(mean_pixel)
                        color_votes[color] = color_votes.get(color, 0) + 1

            if not color_votes:
                return 'unknown'

            return max(color_votes.items(), key=lambda x: x[1])[0]

    filter_obj = DummyFilter()

    # Create a half-black, half-white image (should detect black as dominant)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :50] = [0, 0, 0]  # Black left half
    img[:, 50:] = [255, 255, 255]  # White right half

    detected = filter_obj._get_patchwise_dominant_color(img)
    print(f"Half black, half white image: Detected = {detected}")
    print(f"  Note: Voting-based, so could be either black or white")

    # Create mostly red with some white
    img = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)  # Red background
    img[40:60, 40:60] = [255, 255, 255]  # Small white square in center

    detected = filter_obj._get_patchwise_dominant_color(img)
    print(f"Mostly red with white center: Detected = {detected}")
    print(f"  Expected: red (should be dominant)")

    # Create a black car-like shape on white background
    img = np.full((100, 100, 3), [255, 255, 255], dtype=np.uint8)  # White background
    img[30:70, 20:80] = [0, 0, 0]  # Black "car" in center

    detected = filter_obj._get_patchwise_dominant_color(img)
    print(f"Black rectangle on white background: Detected = {detected}")
    print(f"  Expected: could be black or white depending on area ratio")

    print("=" * 50)


if __name__ == "__main__":
    # Run basic color tests
    success = test_color_detection()

    # Run mixed color tests
    test_mixed_colors()

    if success:
        print("\n✓ All basic tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
