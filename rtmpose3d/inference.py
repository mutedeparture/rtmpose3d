"""
RTMPose3D Inference - Standalone Package
Simple interface: numpy array in -> 3D keypoints out
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)

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
    DETECTOR_CHECKPOINT_URL, POSE_L_CHECKPOINT_URL, POSE_X_CHECKPOINT_URL
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
            logger.info("Downloading detector checkpoint...")
            detector_checkpoint = get_checkpoint_path(DETECTOR_CHECKPOINT_URL, cache_dir)
        elif detector_checkpoint.startswith('http'):
            detector_checkpoint = get_checkpoint_path(detector_checkpoint, cache_dir)
            
        if pose_checkpoint is None:
            # Auto-download pose checkpoint from HuggingFace Hub
            pose_url = POSE_L_CHECKPOINT_URL if self.model_size == 'l' else POSE_X_CHECKPOINT_URL
            if pose_url:
                logger.info("Downloading pose checkpoint...")
                pose_checkpoint = get_checkpoint_path(pose_url, cache_dir)
            else:
                raise RuntimeError(
                    f"No checkpoint URL configured for model size '{self.model_size}'.\n"
                    f"Supported model sizes: 'l' (large), 'x' (extra large)"
                )
        elif pose_checkpoint.startswith('http'):
            pose_checkpoint = get_checkpoint_path(pose_checkpoint, cache_dir)
        
        logger.info("Initializing models...")
        
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
        
        logger.info("Models loaded successfully!")
    
    def __call__(
        self,
        image: np.ndarray,
        bbox_thr: float = 0.3,
        single_person: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run 3D pose estimation on a numpy array image
        
        Args:
            image: Numpy array (HWC, BGR format)
            bbox_thr: Bounding box confidence threshold
            single_person: If True, only detect the most prominent person (largest bbox)
            
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
        bbox_scores = pred_instances.scores[mask].cpu().numpy()
        labels = pred_instances.labels[mask].cpu().numpy()
        
        # Filter for person class (label 0 in COCO)
        person_mask = labels == 0
        bboxes = bboxes[person_mask]
        bbox_scores = bbox_scores[person_mask]
        
        # If single_person mode, keep only the most prominent person
        if single_person and len(bboxes) > 0:
            # Calculate bbox areas (width * height)
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            # Use combination of area and confidence to find most prominent
            prominence = areas * bbox_scores
            most_prominent_idx = np.argmax(prominence)
            bboxes = bboxes[most_prominent_idx:most_prominent_idx+1]
            logger.info(f"Single person mode: selected person with bbox area {areas[most_prominent_idx]:.0f}px, confidence {bbox_scores[most_prominent_idx]:.3f}")
        
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
            
            # For 2D, use the transformed (projected) keypoints from MMPose
            # These are the 2D pixel coordinates in the image space
            transformed_kpts = pred.transformed_keypoints
            if hasattr(transformed_kpts, 'cpu'):
                transformed_kpts = transformed_kpts.cpu().numpy()
            # Handle shape: squeeze extra dimensions [1, K, 2] -> [K, 2]
            while transformed_kpts.ndim > 2 and transformed_kpts.shape[0] == 1:
                transformed_kpts = np.squeeze(transformed_kpts, axis=0)
            keypoints_2d_list.append(transformed_kpts)
            
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
