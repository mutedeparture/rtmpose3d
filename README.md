# RTMPose3D - Standalone Package

**Dead simple 3D pose estimation: numpy array in → 3D keypoints out**

Standalone PyTorch package for RTMPose3D with automatic checkpoint download and caching.

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

# Install package
pip install -e .
```

**Note:** Model checkpoints (316MB total) will auto-download from GitHub Releases on first use and are cached to `~/.cache/rtmpose3d/checkpoints/`.

## Quick Start

```python
import cv2
from rtmpose3d import RTMPose3DInference

# Initialize (auto-downloads detector checkpoint on first use)
model = RTMPose3DInference(device='cuda:0')

# Run inference
image = cv2.imread('person.jpg')
results = model(image)

# Use results
print(results['keypoints_3d'].shape)  # [N, 133, 3]
```

## Features

- ✅ **Zero-config inference**: Just import and run
- ✅ **Auto-download**: Detector checkpoint downloads automatically to `~/.cache/rtmpose3d/`
- ✅ **Local checkpoints**: Supports local RTMPose3D checkpoint files
- ✅ **Simple API**: Numpy array in → dict with 3D keypoints out
- ✅ **Batch support**: Process multiple people in one image
- ✅ **High accuracy**: Uses PyTorch models (not ONNX) for best results

## Checkpoint Management

The package handles two checkpoints (both auto-download from GitHub Releases):

1. **Detector (RTMDet-M)**: Auto-downloads (~95MB) on first use
2. **Pose (RTMW3D-L)**: Auto-downloads (~221MB) on first use

**Total download: ~316MB** (one-time, then cached)

Cache location: `~/.cache/rtmpose3d/checkpoints/`

To clear cache:
```python
from rtmpose3d import clear_cache
clear_cache()
```

## Output Format

```python
{
    'keypoints_3d': np.ndarray,  # Shape: [N, 133, 3] - XYZ coordinates
    'keypoints_2d': np.ndarray,  # Shape: [N, 133, 2] - XY projection
    'scores': np.ndarray,        # Shape: [N, 133] - confidence scores
    'bboxes': np.ndarray         # Shape: [N, 4] - detection boxes
}
```

## Advanced Usage

### Model Size Selection

```python
# Large model (default, best accuracy)
model = RTMPose3DInference(model_size='l')

# Extra large model (higher accuracy, slower)
model = RTMPose3DInference(model_size='x')
```

### Detection Threshold

```python
# Lower threshold = more detections (may include false positives)
results = model(image, bbox_thr=0.1)

# Higher threshold = fewer detections (more conservative)
results = model(image, bbox_thr=0.5)
```

### Custom Cache Directory

```python
model = RTMPose3DInference(
    cache_dir='/path/to/custom/cache'
)
```

### Using Custom Checkpoints

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
├── __init__.py          # Package entry point
├── inference.py         # Main RTMPose3DInference class
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
    └── downloader.py   # Auto-download utility
```

## Model Info

### RTMW3D-L (Large) - Default
- **Parameters**: ~65M
- **Input**: RGB image (any size, auto-resized)
- **Output**: 133 3D keypoints per person
- **Training**: Cocktail14 dataset

### RTMW3D-X (Extra Large)
- **Parameters**: ~98M  
- **Slightly higher accuracy**
- **Slower inference**

## Cache Location

Models are cached at:
```
~/.cache/rtmpose3d/checkpoints/
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (recommended for GPU acceleration)
- MMPose >= 1.0.0
- MMDetection >= 3.0.0
- MMCV >= 2.0.0

## Citation

```bibtex
@misc{RTMPose3D,
  title={RTMPose3D: Real-Time Multi-Person 3D Pose Estimation},
  author={Bahadir Arac},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/mutedeparture/rtmpose3d}}
}
```

## License

Apache 2.0

## Acknowledgments

Based on [MMPose](https://github.com/open-mmlab/mmpose) by OpenMMLab.
