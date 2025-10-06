"""
Test script for RTMPose3D inference with Transformers-style API.

This script demonstrates:
1. Loading model with from_pretrained()
2. Running inference on an image
3. Validating output shapes and coordinate ranges
4. Testing both NumPy and PyTorch tensor outputs
"""

import cv2
import numpy as np
import sys
from pathlib import Path

try:
    from rtmpose3d import RTMPose3D
except ImportError:
    print("Error: rtmpose3d package not found. Please install it first.")
    sys.exit(1)


def test_inference(image_path: str, device: str = 'cuda:0'):
    """
    Test RTMPose3D inference on a single image.
    
    Args:
        image_path: Path to input image
        device: Device to run inference on ('cuda:0', 'cpu', etc.)
    """
    print("=" * 70)
    print("RTMPose3D Inference Test")
    print("=" * 70)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        print("Please provide a valid image path as argument.")
        return
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Load model
    print(f"\nLoading model on {device}...")
    try:
        model = RTMPose3D.from_pretrained('rbarac/rtmpose3d', device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Model loaded successfully")
    
    # Test with NumPy output
    print("\n" + "-" * 70)
    print("Testing with return_tensors='np'")
    print("-" * 70)
    
    try:
        results_np = model(image, return_tensors='np')
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    # Validate outputs
    n_persons = len(results_np['keypoints_2d'])
    print(f"\nDetected {n_persons} person(s)")
    
    if n_persons > 0:
        kpts_2d = results_np['keypoints_2d'][0]
        kpts_3d = results_np['keypoints_3d'][0]
        scores = results_np['scores'][0]
        
        print(f"\nOutput shapes:")
        print(f"  keypoints_2d: {results_np['keypoints_2d'].shape}")
        print(f"  keypoints_3d: {results_np['keypoints_3d'].shape}")
        print(f"  scores: {results_np['scores'].shape}")
        print(f"  bboxes: {results_np['bboxes'].shape}")
        
        print(f"\n2D Keypoint ranges (person 1):")
        print(f"  X: [{kpts_2d[:, 0].min():.1f}, {kpts_2d[:, 0].max():.1f}] pixels")
        print(f"  Y: [{kpts_2d[:, 1].min():.1f}, {kpts_2d[:, 1].max():.1f}] pixels")
        
        print(f"\n3D Keypoint ranges (person 1):")
        print(f"  X: [{kpts_3d[:, 0].min():.2f}, {kpts_3d[:, 0].max():.2f}] meters")
        print(f"  Y: [{kpts_3d[:, 1].min():.2f}, {kpts_3d[:, 1].max():.2f}] meters")
        print(f"  Z: [{kpts_3d[:, 2].min():.2f}, {kpts_3d[:, 2].max():.2f}] meters")
        
        print(f"\nConfidence scores:")
        print(f"  Mean: {scores.mean():.3f}")
        print(f"  Min: {scores.min():.3f}")
        print(f"  Max: {scores.max():.3f}")
        
        # Validate coordinate ranges
        validation_passed = True
        
        if not (0 <= kpts_2d[:, 0].min() and kpts_2d[:, 0].max() <= w):
            print(f"\n  WARNING: 2D X coordinates out of image bounds [0, {w}]")
            validation_passed = False
        
        if not (0 <= kpts_2d[:, 1].min() and kpts_2d[:, 1].max() <= h):
            print(f"\n  WARNING: 2D Y coordinates out of image bounds [0, {h}]")
            validation_passed = False
        
        if validation_passed:
            print("\n  2D coordinates validation: PASSED")
        
        # Test with PyTorch output
        print("\n" + "-" * 70)
        print("Testing with return_tensors='pt'")
        print("-" * 70)
        
        try:
            results_pt = model(image, return_tensors='pt')
            print(f"\nPyTorch tensor types:")
            print(f"  keypoints_2d: {type(results_pt['keypoints_2d'])}")
            print(f"  keypoints_3d: {type(results_pt['keypoints_3d'])}")
            print(f"  Device: {results_pt['keypoints_2d'].device}")
            print("\n  PyTorch output: PASSED")
        except Exception as e:
            print(f"\n  PyTorch output: FAILED - {e}")
    
    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)


if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python test_inference.py <image_path> [device]")
        print("\nExample:")
        print("  python test_inference.py image.jpg")
        print("  python test_inference.py image.jpg cuda:0")
        print("  python test_inference.py image.jpg cpu")
        sys.exit(1)
    
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
    
    test_inference(image_path, device)
