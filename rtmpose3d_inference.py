"""
RTMPose3D Inference Wrapper
Simple interface: numpy array in -> 3D keypoints out
"""

import sys
import warnings
from pathlib import Path
from typing import Union, Optional, Dict
import numpy as np
import cv2

# Add rtmpose3d to path
RTMPOSE3D_ROOT = Path(__file__).parent
sys.path.insert(0, str(RTMPOSE3D_ROOT))

# Register all mmdet modules first to avoid registry errors
import mmdet.apis
import mmdet.models

from mmpose.apis import init_model as init_pose_model
from mmpose.apis import inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import init_detector, inference_detector

# Import rtmpose3d modules to register them
from rtmpose3d import *  # noqa: F401, F403


class RTMPose3DInference:
    """
    Minimal wrapper for RTMPose3D inference
    
    Example:
        >>> model = RTMPose3DInference()
        >>> image = cv2.imread('image.jpg')  # or any numpy array (HWC, BGR)
        >>> results = model(image)
        >>> print(results['keypoints_3d'].shape)  # [num_persons, num_keypoints, 3]
    """
    
    def __init__(
        self,
        detector_config: Optional[str] = None,
        detector_checkpoint: Optional[str] = None,
        pose_config: Optional[str] = None,
        pose_checkpoint: Optional[str] = None,
        device: str = 'cuda:0'
    ):
        """
        Initialize RTMPose3D inference pipeline
        
        Args:
            detector_config: Path to detector config file
            detector_checkpoint: Path or URL to detector checkpoint
            pose_config: Path to pose estimator config file  
            pose_checkpoint: Path to pose estimator checkpoint
            device: Device to run inference on ('cuda:0', 'cpu', etc.)
        """
        self.device = device
        self.rtmpose3d_root = RTMPOSE3D_ROOT
        
        # Set default paths
        if detector_config is None:
            detector_config = str(self.rtmpose3d_root / 'demo' / 'rtmdet_m_640-8xb32_coco-person.py')
        if detector_checkpoint is None:
            detector_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
        if pose_config is None:
            pose_config = str(self.rtmpose3d_root / 'configs' / 'rtmw3d-l_8xb64_cocktail14-384x288.py')
        if pose_checkpoint is None:
            pose_checkpoint = str(self.rtmpose3d_root / 'demo' / 'rtmw3d-l_cock14-0d4ad840_20240422.pth')
            
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
                - 'keypoints_3d': 3D keypoint coordinates [N, K, 3]
                - 'keypoints_2d': 2D keypoint coordinates [N, K, 2]
                - 'scores': Keypoint confidence scores [N, K]
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

