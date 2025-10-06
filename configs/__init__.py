"""
Configuration files for RTMPose3D models
"""

import os
from pathlib import Path

# Get the directory where config files are stored
CONFIG_DIR = Path(__file__).parent

# Model configurations
DETECTOR_CONFIG = str(CONFIG_DIR / 'rtmdet_m_640-8xb32_coco-person.py')
POSE_L_CONFIG = str(CONFIG_DIR / 'rtmw3d-l_8xb64_cocktail14-384x288.py')
POSE_X_CONFIG = str(CONFIG_DIR / 'rtmw3d-x_8xb32_cocktail14-384x288.py')

# Model URLs (hosted by OpenMMLab)
DETECTOR_CHECKPOINT_URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

# Note: The checkpoint is in the rtmpose3d_original/demo folder, not on OpenMMLab server yet
# Users should provide local path or we use the demo checkpoint
POSE_L_CHECKPOINT_URL = None  # Will use local demo checkpoint
POSE_X_CHECKPOINT_URL = None  # Not available yet

__all__ = [
    'CONFIG_DIR',
    'DETECTOR_CONFIG',
    'POSE_L_CONFIG', 
    'POSE_X_CONFIG',
    'DETECTOR_CHECKPOINT_URL',
    'POSE_L_CHECKPOINT_URL',
    'POSE_X_CHECKPOINT_URL'
]
