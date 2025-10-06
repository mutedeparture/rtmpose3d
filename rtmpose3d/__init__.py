"""
RTMPose3D: Real-Time Multi-Person 3D Pose Estimation
Simple interface: numpy array in -> 3D keypoints out
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# Transformers-style API (recommended)
from .modeling import RTMPose3D
from .configuration import RTMPose3DConfig

# Original API (backward compatible)
from .inference import RTMPose3DInference

# Utilities
from .weights import clear_cache, CACHE_DIR

__version__ = '1.0.0'
__all__ = [
    'RTMPose3D',           # Transformers-style model
    'RTMPose3DConfig',     # Configuration class
    'RTMPose3DInference',  # Original inference class
    'clear_cache',
    'CACHE_DIR'
]

