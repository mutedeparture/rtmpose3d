"""
Configuration class for RTMPose3D models (Transformers-style)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any


class RTMPose3DConfig:
    """
    Configuration class for RTMPose3D models.
    
    Similar to transformers.PretrainedConfig, this stores model configuration
    and can be saved/loaded from disk.
    
    Args:
        model_size (str): Model size, either 'l' (large) or 'x' (extra large)
        num_keypoints (int): Number of keypoints to detect (default: 133)
        bbox_thr (float): Bounding box confidence threshold (default: 0.3)
        device (str): Device to run inference on (default: 'cuda:0')
        cache_dir (Optional[str]): Directory to cache checkpoints
        detector_config (Optional[str]): Path to detector config file
        pose_config (Optional[str]): Path to pose config file
        detector_checkpoint (Optional[str]): Path/URL to detector checkpoint
        pose_checkpoint (Optional[str]): Path/URL to pose checkpoint
        **kwargs: Additional keyword arguments
    """
    
    model_type = "rtmpose3d"
    
    def __init__(
        self,
        model_size: str = "l",
        num_keypoints: int = 133,
        bbox_thr: float = 0.3,
        device: str = "cuda:0",
        cache_dir: Optional[str] = None,
        detector_config: Optional[str] = None,
        pose_config: Optional[str] = None,
        detector_checkpoint: Optional[str] = None,
        pose_checkpoint: Optional[str] = None,
        **kwargs
    ):
        self.model_size = model_size.lower()
        self.num_keypoints = num_keypoints
        self.bbox_thr = bbox_thr
        self.device = device
        self.cache_dir = cache_dir
        self.detector_config = detector_config
        self.pose_config = pose_config
        self.detector_checkpoint = detector_checkpoint
        self.pose_checkpoint = pose_checkpoint
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_type": self.model_type,
            "model_size": self.model_size,
            "num_keypoints": self.num_keypoints,
            "bbox_thr": self.bbox_thr,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "detector_config": self.detector_config,
            "pose_config": self.pose_config,
            "detector_checkpoint": self.detector_checkpoint,
            "pose_checkpoint": self.pose_checkpoint,
        }
    
    def save_pretrained(self, save_directory: str):
        """
        Save configuration to directory.
        
        Args:
            save_directory: Directory to save config.json
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        config_file = save_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Configuration saved to {config_file}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load configuration from directory or model name.
        
        Args:
            pretrained_model_name_or_path: Path to directory or model name
            **kwargs: Override config parameters
            
        Returns:
            RTMPose3DConfig instance
        """
        config_file = Path(pretrained_model_name_or_path) / "config.json"
        
        if config_file.exists():
            # Load from file
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            # Override with kwargs
            config_dict.update(kwargs)
            return cls(**config_dict)
        else:
            # Create from model name (e.g., "rtmpose3d-l")
            if "rtmpose3d-l" in pretrained_model_name_or_path.lower():
                model_size = "l"
            elif "rtmpose3d-x" in pretrained_model_name_or_path.lower():
                model_size = "x"
            else:
                model_size = "l"  # default
            
            return cls(model_size=model_size, **kwargs)
    
    def __repr__(self):
        return f"RTMPose3DConfig(model_size='{self.model_size}', num_keypoints={self.num_keypoints}, device='{self.device}')"
