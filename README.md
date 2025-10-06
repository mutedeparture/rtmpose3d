# RTMPose3D - Standalone Package

Standalone PyTorch package for RTMPose3D whole-body 3D pose estimation with automatic checkpoint download and caching. Provides both a simple inference API and a HuggingFace Transformers-style interface.

## Features

- Simple API: numpy array in, 3D keypoints out
- HuggingFace-style interface with `from_pretrained()`
- Automatic checkpoint download from HuggingFace Hub
- 133 keypoints: body (17) + feet (6) + face (68) + hands (42)
- Both 2D and 3D keypoint outputs
- Batch processing support
- Local checkpoint caching

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended)
- 4GB+ VRAM

## Installation

```bash
# Clone the repository
git clone https://github.com/mutedeparture/rtmpose3d.git
cd rtmpose3d

# Install PyTorch (if not already installed)
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install mmcv pre-built wheel (REQUIRED - avoids long compilation)
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# Install other dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

**Note:** Model checkpoints (~330MB total) auto-download from HuggingFace Hub on first use and are cached to `~/.cache/rtmpose3d/checkpoints/`.

## Quick Start

### Option 1: HuggingFace Transformers-Style API (Recommended)

```python
import cv2
from rtmpose3d import RTMPose3D

# Initialize model (auto-downloads from HuggingFace Hub)
model = RTMPose3D.from_pretrained('rbarac/rtmpose3d', device='cuda:0')

# Run inference
image = cv2.imread('person.jpg')
results = model(image, return_tensors='np')

# Access results
keypoints_3d = results['keypoints_3d']  # [N, 133, 3]
keypoints_2d = results['keypoints_2d']  # [N, 133, 2]
scores = results['scores']              # [N, 133]
```

### Option 2: Simple Inference API

```python
import cv2
from rtmpose3d import RTMPose3DInference

# Initialize
model = RTMPose3DInference(device='cuda:0')

# Run inference
image = cv2.imread('person.jpg')
results = model(image)

# Use results
print(results['keypoints_3d'].shape)  # [N, 133, 3]
```

## Output Format

```python
{
    'keypoints_3d': np.ndarray,  # Shape: [N, 133, 3] - 3D coordinates in meters
    'keypoints_2d': np.ndarray,  # Shape: [N, 133, 2] - 2D pixel coordinates
    'scores': np.ndarray,        # Shape: [N, 133] - confidence scores [0, 1]
    'bboxes': np.ndarray         # Shape: [N, 4] - detection boxes [x1, y1, x2, y2]
}
```

Where `N` is the number of detected persons.

## Coordinate Systems

### 2D Keypoints
- Pixel coordinates in the input image space
- **X**: horizontal position [0, image_width]
- **Y**: vertical position [0, image_height]

### 3D Keypoints
- World coordinates in meters, camera-relative
- **X**: horizontal (left-right), negative = left, positive = right
- **Y**: depth (distance from camera), negative = closer, positive = farther
- **Z**: vertical (up-down), negative = down, positive = up

**Note:** The 3D coordinate system uses Z-up convention. Body height is measured along the Z-axis.

## Keypoint Format

The model outputs **133 keypoints** per person in COCO-WholeBody format:

| Index Range | Body Part | Description |
|-------------|-----------|-------------|
| 0-16 | Body | COCO body keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |
| 17-22 | Feet | Foot keypoints |
| 23-90 | Face | 68 facial landmarks |
| 91-111 | Left Hand | 21 left hand keypoints |
| 112-132 | Right Hand | 21 right hand keypoints |

### Body Keypoints (0-16)
```
0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow,
9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip,
13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
```

## Advanced Usage

### HuggingFace Transformers API

#### Return PyTorch Tensors

```python
from rtmpose3d import RTMPose3D

model = RTMPose3D.from_pretrained('rbarac/rtmpose3d', device='cuda:0')
results = model(image, return_tensors='pt')

# Results contain PyTorch tensors on the same device as the model
keypoints_3d = results['keypoints_3d']  # torch.Tensor on cuda:0
```

#### Save and Load Configuration

```python
# Save configuration
model.save_pretrained('./my_model')

# Load from local directory
model = RTMPose3D.from_pretrained('./my_model', device='cuda:0')
```

### Simple Inference API

#### Model Size Selection

```python
# Large model (default, best balance)
model = RTMPose3DInference(model_size='l')

# Extra large model (higher accuracy, slower)
model = RTMPose3DInference(model_size='x')
```

#### Detection Threshold

```python
# Lower threshold = more detections (may include false positives)
results = model(image, bbox_thr=0.1)

# Higher threshold = fewer detections (more conservative)
results = model(image, bbox_thr=0.5)
```

#### Custom Cache Directory

```python
model = RTMPose3DInference(
    cache_dir='/path/to/custom/cache'
)
```

#### Using Custom Checkpoints

```python
model = RTMPose3DInference(
    detector_checkpoint='/path/to/detector.pth',
    pose_checkpoint='/path/to/pose.pth'
)
```

### Clear Downloaded Checkpoints

```python
from rtmpose3d import clear_cache
clear_cache()  # Removes all cached checkpoints
```

## Checkpoint Management

The package handles two checkpoints (auto-download from HuggingFace Hub):

1. **Person Detector (RTMDet-M)**: ~99MB
2. **Pose Estimator (RTMW3D-L)**: ~231MB

**Total download: ~330MB** (one-time, then cached)

Cache location: `~/.cache/rtmpose3d/checkpoints/`

Checkpoints are hosted on HuggingFace Hub: [rbarac/rtmpose3d](https://huggingface.co/rbarac/rtmpose3d)

## Keypoint Format

The model outputs **133 keypoints** (COCO-WholeBody format):

- **0-16**: Body keypoints (COCO format)
- **17-22**: Foot keypoints
- **23-90**: Face keypoints (68 points)
- **91-112**: Left hand keypoints (21 points)
- **113-132**: Right hand keypoints (21 points)

### 3D Coordinate System

- **X**: Left-right (negative = left, positive = right)
- **Y**: Depth (negative = closer, positive = farther)  
- **Z**: Up-down (negative = down, positive = up)

## Examples

See `examples/basic_usage.py` for a complete working example.

```bash
python examples/basic_usage.py
```

## Package Structure

```
rtmpose3d/
├── __init__.py          # Package exports
├── inference.py         # RTMPose3DInference class
├── modeling.py          # RTMPose3D Transformers-style class
├── configuration.py     # RTMPose3DConfig class
├── models/             # Custom RTMPose3D modules
│   ├── __init__.py
│   ├── rtmw3d_head.py
│   ├── pose_estimator.py
│   ├── simcc_3d_label.py
│   ├── loss.py
│   └── utils.py
├── configs/            # Model configurations
│   ├── __init__.py
│   ├── rtmdet_m_640-8xb32_coco-person.py
│   ├── rtmw3d-l_8xb64_cocktail14-384x288.py
│   └── rtmw3d-x_8xb32_cocktail14-384x288.py
└── weights/            # Checkpoint management
    ├── __init__.py
    └── downloader.py   # Auto-download from HuggingFace Hub
```

## Model Information

### RTMW3D-L (Large) - Default
- **Parameters**: ~65M
- **Input**: RGB image (any resolution, auto-resized to 384x288)
- **Output**: 133 3D keypoints per person
- **Training Dataset**: Cocktail14 (14 public datasets combined)
- **Speed**: Real-time on modern GPUs

### RTMW3D-X (Extra Large)
- **Parameters**: ~98M  
- **Accuracy**: Slightly higher than L model
- **Speed**: Slower inference than L model

## Technical Notes

### 2D Keypoint Extraction
2D keypoints are extracted from MMPose's `transformed_keypoints` attribute, which provides pixel coordinates in the original image space. This ensures accurate visualization and alignment with the input image.

### 3D Coordinate Transformation
The raw 3D coordinates from RTMPose3D undergo the transformation `-kpts[..., [0, 2, 1]]` to:
- Swap Y and Z axes (making Z vertical instead of Y)
- Negate all coordinates for conventional orientation

This results in a Z-up coordinate system commonly used in 3D graphics applications.

### Body Height Measurement
Typical height measurements (nose to ankle) range from 1.5 to 3.0 meters depending on:
- Camera perspective and lens distortion
- Person's distance from camera
- Model's coordinate system scale

The relative positions between keypoints are more reliable than absolute measurements.

## Citation

```bibtex
@misc{rtmpose3d2025,
  title={RTMPose3D: Real-Time Multi-Person 3D Pose Estimation},
  author={Arac, Bahadir},
  year={2025},
  publisher={GitHub},
  url={https://github.com/mutedeparture/rtmpose3d}
}
```

## License

Apache 2.0

## Acknowledgments

Built on [MMPose](https://github.com/open-mmlab/mmpose) by OpenMMLab. RTMPose3D models trained by the MMPose team.

````
