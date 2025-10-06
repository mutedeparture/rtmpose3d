#!/usr/bin/env python3
"""
Basic usage example for RTMPose3D standalone package.

Usage:
    python basic_usage.py <image_path>

Example:
    python basic_usage.py person.jpg
"""

import cv2
import sys
from rtmpose3d import RTMPose3DInference


def main():
    print("=" * 60)
    print("RTMPose3D Standalone Package - Basic Usage")
    print("=" * 60)
    
    # Check for image path argument
    if len(sys.argv) < 2:
        print("\nUsage: python basic_usage.py <image_path>")
        print("Example: python basic_usage.py person.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # 1. Initialize model (auto-downloads checkpoints on first run)
    print("\n1. Initializing model...")
    model = RTMPose3DInference(
        model_size='l',  # 'l' for large, 'x' for extra large
        device='cuda:0'   # or 'cpu'
    )
    
    # 2. Load image
    print("\n2. Loading image...")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        sys.exit(1)
    
    print(f"   Image shape: {image.shape}")
    
    # 3. Run inference
    print("\n3. Running inference...")
    results = model(image, bbox_thr=0.1)
    
    # 4. Print results
    print("\n4. Results:")
    print(f"   Detected {len(results['keypoints_3d'])} person(s)")
    
    if len(results['keypoints_3d']) > 0:
        print(f"   3D keypoints shape: {results['keypoints_3d'].shape}")
        print(f"   2D keypoints shape: {results['keypoints_2d'].shape}")
        print(f"   Scores shape: {results['scores'].shape}")
        print(f"   Bboxes shape: {results['bboxes'].shape}")
        
        # Print first person's nose coordinates
        print(f"\n   First person's nose (keypoint 0) 3D coordinates:")
        print(f"     X: {results['keypoints_3d'][0, 0, 0]:.4f}")
        print(f"     Y: {results['keypoints_3d'][0, 0, 1]:.4f}")
        print(f"     Z: {results['keypoints_3d'][0, 0, 2]:.4f}")
        print(f"     Score: {results['scores'][0, 0]:.4f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
