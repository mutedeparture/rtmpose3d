"""
Test single person detection mode.

This script demonstrates the single_person feature which detects
only the most prominent person in the image.

Usage:
    python test_single_person.py <image_path>
"""

import cv2
import sys
from pathlib import Path

try:
    from rtmpose3d import RTMPose3D
except ImportError:
    print("Error: rtmpose3d package not found.")
    sys.exit(1)


def test_single_vs_multi(image_path: str, device: str = 'cuda:0'):
    """
    Compare detection with and without single_person mode.
    
    Args:
        image_path: Path to input image
        device: Device to run inference on
    """
    print("=" * 70)
    print("Single Person Detection Test")
    print("=" * 70)
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image")
        return
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Load model
    print(f"\nLoading model on {device}...")
    model = RTMPose3D.from_pretrained('rbarac/rtmpose3d', device=device)
    
    # Test multi-person detection (default)
    print("\n" + "-" * 70)
    print("Multi-person detection (default)")
    print("-" * 70)
    results_multi = model(image, return_tensors='np')
    n_persons_multi = len(results_multi['keypoints_2d'])
    print(f"Detected {n_persons_multi} person(s)")
    
    if n_persons_multi > 0:
        for i in range(n_persons_multi):
            scores = results_multi['scores'][i]
            bbox = results_multi['bboxes'][i]
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            print(f"  Person {i+1}: bbox area = {bbox_area:.0f}px, avg confidence = {scores.mean():.3f}")
    
    # Test single-person detection
    print("\n" + "-" * 70)
    print("Single-person detection (single_person=True)")
    print("-" * 70)
    results_single = model(image, single_person=True, return_tensors='np')
    n_persons_single = len(results_single['keypoints_2d'])
    print(f"Detected {n_persons_single} person(s)")
    
    if n_persons_single > 0:
        scores = results_single['scores'][0]
        bbox = results_single['bboxes'][0]
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        print(f"  Selected person: bbox area = {bbox_area:.0f}px, avg confidence = {scores.mean():.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Multi-person mode:  {n_persons_multi} person(s) detected")
    print(f"Single-person mode: {n_persons_single} person(s) detected (most prominent)")
    
    if n_persons_multi > 1:
        print(f"\nNote: single_person=True selected the most prominent person from {n_persons_multi} detections")
        print("      based on bbox area × detection confidence")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_single_person.py <image_path> [device]")
        print("\nExample:")
        print("  python test_single_person.py image.jpg")
        print("  python test_single_person.py image.jpg cuda:0")
        sys.exit(1)
    
    image_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
    
    test_single_vs_multi(image_path, device)
