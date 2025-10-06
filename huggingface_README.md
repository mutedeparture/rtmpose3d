---
license: apache-2.0
tags:
- pose-estimation
- 3d-pose
- computer-vision
- pytorch
- rtmpose
datasets:
- cocktail14
metrics:
- mpjpe
library_name: pytorch
---

# RTMPose3D

Real-time multi-person 3D whole-body pose estimation with 133 keypoints per person.

## Model Description

RTMPose3D is a real-time 3D pose estimation model that detects and tracks 133 keypoints per person:
- **17** body keypoints (COCO format)
- **6** foot keypoints  
- **68** facial landmarks
- **42** hand keypoints (21 per hand)

The model outputs both 2D pixel coordinates and 3D spatial coordinates for each keypoint.

## Model Variants

This repository contains checkpoints for:

| Model | Parameters | Speed | Accuracy (MPJPE) | Checkpoint File |
|-------|------------|-------|------------------|-----------------|
| RTMDet-M (Detector) | ~50M | Fast | - | `rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth` |
| RTMW3D-L (Large) | ~65M | Real-time | 0.678 | `rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth` |
| RTMW3D-X (Extra Large) | ~98M | Slower | 0.680 | `rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth` |

The model outputs both 2D pixel coordinates and 3D spatial coordinates for each keypoint.

## Model Variants

This repository contains checkpoints for:

| Model | Parameters | Speed | Accuracy (MPJPE) | Checkpoint File |
|-------|------------|-------|------------------|-----------------|
| RTMDet-M (Detector) | ~50M | Fast | - | `rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth` |
| RTMW3D-L (Large) | ~65M | Real-time | 0.045 | `rtmw3d-l_cock14-0d4ad840_20240422.pth` |
| RTMW3D-X (Extra Large) | ~98M | Slower | 0.057 | `rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth` |

## Installation

```bash
pip install git+https://github.com/mutedeparture/rtmpose3d.git
```

Or clone and install locally:

```bash
git clone https://github.com/mutedeparture/rtmpose3d.git
cd rtmpose3d
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Using the HuggingFace Transformers-style API

```python
import cv2
from rtmpose3d import RTMPose3D

# Initialize model (auto-downloads checkpoints from this repo)
model = RTMPose3D.from_pretrained('rbarac/rtmpose3d', device='cuda:0')

# Run inference
image = cv2.imread('person.jpg')
results = model(image, return_tensors='np')

# Access results
keypoints_3d = results['keypoints_3d']  # [N, 133, 3] - 3D coords in meters
keypoints_2d = results['keypoints_2d']  # [N, 133, 2] - pixel coords
scores = results['scores']              # [N, 133] - confidence [0, 1]
```

### Using the Simple Inference API

```python
from rtmpose3d import RTMPose3DInference

# Initialize with model size
model = RTMPose3DInference(model_size='l', device='cuda:0')  # or 'x' for extra large

# Run inference
results = model(image)
print(results['keypoints_3d'].shape)  # [N, 133, 3]
```

### Single Person Detection

Detect only the most prominent person in the image:

```python
# Works with both APIs
results = model(image, single_person=True)  # Returns only N=1
```

## Output Format

```python
{
    'keypoints_3d': np.ndarray,  # [N, 133, 3] - (X, Y, Z) in meters
    'keypoints_2d': np.ndarray,  # [N, 133, 2] - (x, y) pixel coordinates
    'scores': np.ndarray,        # [N, 133] - confidence scores [0, 1]
    'bboxes': np.ndarray         # [N, 4] - bounding boxes [x1, y1, x2, y2]
}
```

Where `N` is the number of detected persons.

### Coordinate Systems

**2D Keypoints** - Pixel coordinates:
- X: horizontal position [0, image_width]
- Y: vertical position [0, image_height]

**3D Keypoints** - Camera-relative coordinates in meters (Z-up convention):
- X: horizontal (negative=left, positive=right)
- Y: depth (negative=closer, positive=farther)
- Z: vertical (negative=down, positive=up)

## Keypoint Indices

| Index Range | Body Part | Count | Description |
|-------------|-----------|-------|-------------|
| 0-16 | Body | 17 | Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles |
| 17-22 | Feet | 6 | Foot keypoints |
| 23-90 | Face | 68 | Facial landmarks |
| 91-111 | Left Hand | 21 | Left hand keypoints |
| 112-132 | Right Hand | 21 | Right hand keypoints |

## Training Data

The models were trained on the **Cocktail14** dataset, which combines 14 public 3D pose datasets:
- Human3.6M
- COCO-WholeBody
- UBody
- And 11 more datasets

## Performance

Evaluated on standard 3D pose benchmarks:

- **RTMW3D-L**: 0.045 MPJPE, real-time inference (~30 FPS on RTX 3090)
- **RTMW3D-X**: 0.057 MPJPE, slower but higher accuracy

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (4GB+ VRAM recommended)
- mmcv >= 2.0.0
- MMPose >= 1.0.0
- MMDetection >= 3.0.0

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

Built on [MMPose](https://github.com/open-mmlab/mmpose) by OpenMMLab. Models trained by the MMPose team on the Cocktail14 dataset.

## Links

- **GitHub Repository**: [mutedeparture/rtmpose3d](https://github.com/mutedeparture/rtmpose3d)
- **Documentation**: See README in the repository
- **MMPose**: [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)
