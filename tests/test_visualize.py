"""
Visualization test for RTMPose3D keypoints.

This script:
1. Runs inference on an image
2. Visualizes 2D keypoints overlaid on the image
3. Saves the annotated result

Usage:
    python test_visualization.py <image_path> [output_path]

Example:
    python test_visualization.py person.jpg annotated.jpg
"""

import cv2
import numpy as np
import sys
from pathlib import Path

try:
    from rtmpose3d import RTMPose3D
except ImportError:
    print("Error: rtmpose3d package not found.")
    sys.exit(1)


def draw_keypoints(image, keypoints_2d, scores, threshold=0.3):
    """
    Draw keypoints on image.
    
    Args:
        image: Input image (BGR)
        keypoints_2d: 2D keypoints array [N, 2]
        scores: Confidence scores [N]
        threshold: Minimum confidence threshold
        
    Returns:
        Annotated image
    """
    img = image.copy()
    
    # Define colors for different body parts
    colors = {
        'body': (0, 255, 0),      # Green
        'feet': (255, 255, 0),     # Cyan
        'face': (0, 165, 255),     # Orange
        'left_hand': (255, 0, 0),  # Blue
        'right_hand': (0, 0, 255)  # Red
    }
    
    # Draw keypoints
    for i, (x, y) in enumerate(keypoints_2d):
        if scores[i] < threshold:
            continue
            
        # Choose color based on keypoint index
        if i <= 16:
            color = colors['body']
            radius = 4
        elif i <= 22:
            color = colors['feet']
            radius = 4
        elif i <= 90:
            color = colors['face']
            radius = 2
        elif i <= 111:
            color = colors['left_hand']
            radius = 2
        else:
            color = colors['right_hand']
            radius = 2
        
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    
    # Draw skeleton for body keypoints
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    for i, j in skeleton:
        if scores[i] > threshold and scores[j] > threshold:
            pt1 = (int(keypoints_2d[i, 0]), int(keypoints_2d[i, 1]))
            pt2 = (int(keypoints_2d[j, 0]), int(keypoints_2d[j, 1]))
            cv2.line(img, pt1, pt2, colors['body'], 2)
    
    return img


def visualize(image_path: str, output_path: str = None, device: str = 'cuda:0'):
    """
    Visualize keypoints on image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save annotated image (optional)
        device: Device for inference
    """
    print("=" * 70)
    print("RTMPose3D Visualization Test")
    print("=" * 70)
    
    # Load image
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image")
        return
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Load model and run inference
    print(f"\nLoading model on {device}...")
    model = RTMPose3D.from_pretrained('rbarac/rtmpose3d', device=device)
    
    print("Running inference...")
    results = model(image, return_tensors='np')
    
    n_persons = len(results['keypoints_2d'])
    print(f"Detected {n_persons} person(s)")
    
    if n_persons == 0:
        print("No persons detected in image")
        return
    
    # Annotate image for each person
    annotated = image.copy()
    for i in range(n_persons):
        kpts_2d = results['keypoints_2d'][i]
        scores = results['scores'][i]
        annotated = draw_keypoints(annotated, kpts_2d, scores)
        print(f"\nPerson {i+1}:")
        print(f"  Keypoints: {len(kpts_2d)}")
        print(f"  Avg confidence: {scores.mean():.3f}")
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"\nSaved annotated image to: {output_path}")
    else:
        # Create output in same directory as input
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_annotated.jpg"
        cv2.imwrite(str(output_path), annotated)
        print(f"\nSaved annotated image to: {output_path}")
    
    print("\n" + "=" * 70)
    print("Visualization complete")
    print("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_visualization.py <image_path> [output_path]")
        print("\nExample:")
        print("  python test_visualization.py person.jpg")
        print("  python test_visualization.py person.jpg annotated.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualize(image_path, output_path)
