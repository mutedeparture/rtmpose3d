"""
RTMPose3D: Real-Time Multi-Person 3D Pose Estimation
Simple interface: numpy array in -> 3D keypoints out
"""

from .inference import RTMPose3DInference
from .weights import clear_cache, CACHE_DIR

__version__ = '1.0.0'
__all__ = ['RTMPose3DInference', 'clear_cache', 'CACHE_DIR']

