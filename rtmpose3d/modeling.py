"""
RTMPose3D Model - Transformers-style API
"""

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any

from .configuration import RTMPose3DConfig
from .inference import RTMPose3DInference


logger = logging.getLogger(__name__)


class RTMPose3D:
    """
    RTMPose3D model with Transformers-style API.
    
    This class provides a HuggingFace Transformers-style interface for RTMPose3D,
    making it familiar to users of the transformers library.
    
    Example:
        >>> from rtmpose3d import RTMPose3D
        >>> model = RTMPose3D.from_pretrained("rtmpose3d-l", device="cuda:0")
        >>> import cv2
        >>> image = cv2.imread("person.jpg")
        >>> outputs = model(image)
        >>> print(outputs['keypoints_3d'].shape)  # (N, 133, 3)
    """
    
    def __init__(self, config: RTMPose3DConfig):
        """
        Initialize RTMPose3D model with configuration.
        
        Args:
            config: RTMPose3DConfig instance
        """
        self.config = config
        
        # Initialize the underlying inference engine
        self._model = RTMPose3DInference(
            model_size=config.model_size,
            device=config.device,
            cache_dir=config.cache_dir,
            detector_config=config.detector_config,
            pose_config=config.pose_config,
            detector_checkpoint=config.detector_checkpoint,
            pose_checkpoint=config.pose_checkpoint,
        )
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> "RTMPose3D":
        """
        Load pretrained model (Transformers-style).
        
        Args:
            pretrained_model_name_or_path: Model name or path to saved model
                - "rtmpose3d-l" or "rtmpose3d-large": Large model
                - "rtmpose3d-x" or "rtmpose3d-xlarge": Extra large model
                - Path to directory with saved model
            **kwargs: Override config parameters (device, bbox_thr, etc.)
        
        Returns:
            RTMPose3D instance
            
        Example:
            >>> model = RTMPose3D.from_pretrained("rtmpose3d-l", device="cuda:0")
            >>> model = RTMPose3D.from_pretrained("./my-saved-model")
        """
        # Load or create config
        config = RTMPose3DConfig.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        
        # Create model instance
        return cls(config)
    
    def save_pretrained(self, save_directory: str):
        """
        Save model configuration to directory (Transformers-style).
        
        Note: Checkpoints are not saved, only configuration.
        Users can always re-download checkpoints using from_pretrained().
        
        Args:
            save_directory: Directory to save config.json
            
        Example:
            >>> model.save_pretrained("./my-model")
        """
        self.config.save_pretrained(save_directory)
        logger.info(f"Model configuration saved to {save_directory}")
        logger.info("Note: Checkpoints will be auto-downloaded when loading with from_pretrained()")
    
    def __call__(
        self,
        image: np.ndarray,
        bbox_thr: Optional[float] = None,
        single_person: bool = False,
        return_tensors: str = "np"
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Run inference on image (Transformers-style).
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            bbox_thr: Bounding box confidence threshold (uses config.bbox_thr if None)
            single_person: If True, only detect the most prominent person (largest bbox)
            return_tensors: Format of returned tensors
                - "np": Return numpy arrays (default)
                - "pt": Return PyTorch tensors
        
        Returns:
            Dictionary containing:
                - keypoints_3d: 3D keypoints (N, 133, 3) - XYZ coordinates
                - keypoints_2d: 2D keypoints (N, 133, 2) - XY projection
                - scores: Confidence scores (N, 133)
                - bboxes: Detection bounding boxes (N, 4)
                
        Example:
            >>> outputs = model(image)  # numpy arrays, all persons
            >>> outputs = model(image, single_person=True)  # only most prominent person
            >>> outputs = model(image, return_tensors="pt")  # torch tensors
        """
        # Use config bbox_thr if not specified
        if bbox_thr is None:
            bbox_thr = self.config.bbox_thr
        
        # Run inference using underlying model
        results = self._model(image, bbox_thr=bbox_thr, single_person=single_person)
        
        # Convert to requested tensor format
        if return_tensors == "pt":
            results = {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in results.items()
            }
        elif return_tensors != "np":
            raise ValueError(f"return_tensors must be 'np' or 'pt', got '{return_tensors}'")
        
        return results
    
    def forward(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Alias for __call__ (Transformers-style).
        
        Args:
            image: Input image
            **kwargs: Additional arguments passed to __call__
            
        Returns:
            Model outputs
        """
        return self(image, **kwargs)
    
    @property
    def device(self) -> str:
        """Get the device model is running on."""
        return self.config.device
    
    def to(self, device: str) -> "RTMPose3D":
        """
        Move model to device (PyTorch-style).
        
        Args:
            device: Device name (e.g., 'cuda:0', 'cpu')
            
        Returns:
            self (for chaining)
            
        Example:
            >>> model = model.to('cuda:1')
        """
        self.config.device = device
        self._model.device = device
        # Note: Actual model moving happens in the underlying MMPose models
        # This is just for API compatibility
        return self
    
    def __repr__(self):
        return (
            f"RTMPose3D(\n"
            f"  model_size='{self.config.model_size}',\n"
            f"  num_keypoints={self.config.num_keypoints},\n"
            f"  device='{self.config.device}'\n"
            f")"
        )
