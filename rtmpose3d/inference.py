"""
RTMPose3D Inference - Standalone Package
Simple interface: numpy array in -> 3D keypoints out
"""

import sys
import warnings
from pathlib import Path
from typing import Optional, Dict
import numpy as np

# Patch MMDetection version check BEFORE importing mmdet
def _patch_mmdet_version_check():
    """Automatically patch MMDetection to accept mmcv 2.2.0"""
    try:
        # Find mmdet package location
        import importlib.util
        spec = importlib.util.find_spec('mmdet')
        if spec and spec.origin:
            mmdet_init_file = Path(spec.origin)
            if mmdet_init_file.exists():
                content = mmdet_init_file.read_text()
                if "mmcv_maximum_version = '2.2.0'" in content:
                    content = content.replace(
                        "mmcv_maximum_version = '2.2.0'",
                        "mmcv_maximum_version = '2.3.0'"
                    )
                    mmdet_init_file.write_text(content)
    except Exception as e:
        warnings.warn(f"Failed to patch MMDetection version check: {e}")

# Apply patch before any mmdet imports
_patch_mmdet_version_check()

# Add package models to path for registration
PACKAGE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PACKAGE_ROOT))

# Register mmdet and mmpose modules
import mmdet.apis
import mmdet.models
from mmpose.apis import init_model as init_pose_model
from mmpose.apis import inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import init_detector, inference_detector

# Import rtmpose3d custom modules to register them
from . import models  # noqa: F401

# Import config paths and checkpoint URLs
from .configs import (
    DETECTOR_CONFIG, POSE_L_CONFIG, POSE_X_CONFIG,
    DETECTOR_CHECKPOINT_URL, POSE_L_CHECKPOINT_URL
)
from .weights import get_checkpoint_path


class RTMPose3DInference:
    """
    Standalone RTMPose3D inference wrapper
    
    Automatically downloads and caches model weights on first use.
    
    Example:
        >>> from rtmpose3d import RTMPose3DInference
        >>> import cv2
        >>>
        >>> # Initialize (downloads models if needed)
        >>> model = RTMPose3DInference(device='cuda:0')
        >>>
        >>> # Inference
        >>> image = cv2.imread('person.jpg')
        >>> results = model(image)
        >>>
        >>> # Results
        >>> print(results['keypoints_3d'].shape)  # [N, 133, 3]
    """
    
    def __init__(
        self,
        model_size: str = 'l',
        detector_config: Optional[str] = None,
        detector_checkpoint: Optional[str] = None,
        pose_config: Optional[str] = None,
        pose_checkpoint: Optional[str] = None,
        device: str = 'cuda:0',
        cache_dir: Optional[str] = None
    ):
        """
        Initialize RTMPose3D inference pipeline
        
        Args:
            model_size: Model size ('l' for large, 'x' for extra large)
            detector_config: Path to detector config (uses default if None)
            detector_checkpoint: Path/URL to detector checkpoint (auto-downloads if None)
            pose_config: Path to pose config (uses default if None)  
            pose_checkpoint: Path to pose checkpoint (auto-downloads if None)
            device: Device to run on ('cuda:0', 'cpu', etc.)
            cache_dir: Directory to cache downloaded checkpoints
        """
        self.device = device
        self.model_size = model_size.lower()
        
        # Use default configs from package
        if detector_config is None:
            detector_config = DETECTOR_CONFIG
        if pose_config is None:
            pose_config = POSE_L_CONFIG if self.model_size == 'l' else POSE_X_CONFIG
        
        # Auto-download checkpoints if URLs provided
        if detector_checkpoint is None:
            print("📥 Downloading detector checkpoint...")
            detector_checkpoint = get_checkpoint_path(DETECTOR_CHECKPOINT_URL, cache_dir)
        elif detector_checkpoint.startswith('http'):
            detector_checkpoint = get_checkpoint_path(detector_checkpoint, cache_dir)
            
        if pose_checkpoint is None:
            # Auto-download pose checkpoint from GitHub releases
            pose_url = POSE_L_CHECKPOINT_URL if self.model_size == 'l' else None
            if pose_url:
                print("📥 Downloading pose checkpoint...")
                pose_checkpoint = get_checkpoint_path(pose_url, cache_dir)
            else:
                # Fallback: try local checkpoint from rtmpose3d_original if available
                original_checkpoint = Path(__file__).parent.parent / 'rtmpose3d_original' / 'demo' / 'rtmw3d-l_cock14-0d4ad840_20240422.pth'
                if original_checkpoint.exists():
                    print(f"📦 Using local checkpoint: {original_checkpoint}")
                    pose_checkpoint = str(original_checkpoint)
                else:
                    # Try cached version
                    cached_checkpoint = Path.home() / '.cache' / 'rtmpose3d' / 'checkpoints' / 'rtmw3d-l_cock14-0d4ad840_20240422.pth'
                    if cached_checkpoint.exists():
                        print(f"✓ Using cached checkpoint: {cached_checkpoint}")
                        pose_checkpoint = str(cached_checkpoint)
                    else:
                        raise RuntimeError(
                            f"RTMPose3D checkpoint not found and no download URL available.\n"
                            f"Please provide pose_checkpoint parameter or download manually:\n"
                            f"  Expected at: {cached_checkpoint}\n"
                            f"  Or from: https://github.com/mutedeparture/rtmpose3d/releases"
                        )
        elif pose_checkpoint.startswith('http'):
            pose_checkpoint = get_checkpoint_path(pose_checkpoint, cache_dir)
        
        print("🔧 Initializing models...")
        
        # Initialize detector
        self.detector = init_detector(
            detector_config,
            detector_checkpoint,
            device=device
        )
        # Fix mmdet pipeline registry issue
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        
        # Initialize pose estimator
        self.pose_estimator = init_pose_model(
            pose_config,
            pose_checkpoint,
            device=device
        )
        
        print("✅ Models loaded successfully!")
    
    def __call__(
        self,
        image: np.ndarray,
        bbox_thr: float = 0.3
    ) -> Dict[str, np.ndarray]:
        """
        Run 3D pose estimation on a numpy array image
        
        Args:
            image: Numpy array (HWC, BGR format)
            bbox_thr: Bounding box confidence threshold
            
        Returns:
            Dictionary with numpy arrays:
                - 'keypoints_3d': 3D keypoint coordinates [N, 133, 3]
                - 'keypoints_2d': 2D keypoint coordinates [N, 133, 2]
                - 'scores': Keypoint confidence scores [N, 133]
                - 'bboxes': Detected person bboxes [N, 4]
        """
        # Detect persons
        det_results = inference_detector(self.detector, image)
        pred_instances = det_results.pred_instances
        
        # Filter by bbox score and convert to numpy
        mask = pred_instances.scores > bbox_thr
        bboxes = pred_instances.bboxes[mask].cpu().numpy()
        labels = pred_instances.labels[mask].cpu().numpy()
        
        # Filter for person class (label 0 in COCO)
        person_mask = labels == 0
        bboxes = bboxes[person_mask]
        
        if len(bboxes) == 0:
            warnings.warn(f"No person detected in image (found {len(pred_instances.bboxes)} total detections, {mask.sum()} above threshold {bbox_thr})")
            return {
                'keypoints_3d': np.array([]),
                'keypoints_2d': np.array([]),
                'scores': np.array([]),
                'bboxes': np.array([])
            }
        
        # Run pose estimation
        pose_results = inference_topdown(self.pose_estimator, image, bboxes)
        
        # Extract results
        keypoints_3d_list = []
        keypoints_2d_list = []
        scores_list = []
        
        for i, result in enumerate(pose_results):
            pred = result.pred_instances
            
            # Get keypoints (these are 3D for RTMPose3D models)
            kpts = pred.keypoints
            if hasattr(kpts, 'cpu'):
                kpts = kpts.cpu().numpy()
            
            # Handle shape: squeeze extra dimensions [1, 1, K, 3] -> [K, 3]
            while kpts.ndim > 2 and kpts.shape[0] == 1:
                kpts = np.squeeze(kpts, axis=0)
            
            # RTMPose3D outputs 3D keypoints directly
            # Transform coordinates: -x, -z, -y to match visualization convention
            keypoints_3d = -kpts[..., [0, 2, 1]]
            keypoints_3d_list.append(keypoints_3d)
            
            # For 2D, project 3D keypoints to XY plane
            keypoints_2d = kpts[..., :2]
            keypoints_2d_list.append(keypoints_2d)
            
            # Get scores
            scores = pred.keypoint_scores
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            # Squeeze extra dimensions [1, 1, K] -> [K]
            while scores.ndim > 1 and scores.shape[0] == 1:
                scores = np.squeeze(scores, axis=0)
            scores_list.append(scores)
        
        return {
            'keypoints_3d': np.stack(keypoints_3d_list) if keypoints_3d_list else np.array([]),
            'keypoints_2d': np.stack(keypoints_2d_list) if keypoints_2d_list else np.array([]),
            'scores': np.stack(scores_list) if scores_list else np.array([]),
            'bboxes': bboxes
        }
