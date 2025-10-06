"""
Automatic checkpoint downloader for RTMPose3D models
"""

import os
import logging
from pathlib import Path
from typing import Optional
import hashlib
from urllib.request import urlretrieve
from tqdm import tqdm


logger = logging.getLogger(__name__)


class DownloadProgressBar:
    """Progress bar for downloads"""
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading')
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


# Default cache directory
CACHE_DIR = Path.home() / '.cache' / 'rtmpose3d' / 'checkpoints'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_checkpoint_path(url: str, cache_dir: Optional[Path] = None) -> str:
    """
    Download checkpoint if not cached, return local path
    
    Args:
        url: URL to checkpoint file
        cache_dir: Optional cache directory (default: ~/.cache/rtmpose3d/checkpoints)
    
    Returns:
        str: Local path to checkpoint file
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from URL
    filename = url.split('/')[-1]
    local_path = cache_dir / filename
    
    # Check if already downloaded
    if local_path.exists():
        logger.info(f"Using cached checkpoint: {local_path}")
        return str(local_path)
    
    # Download
    logger.info(f"Downloading checkpoint from {url}")
    logger.info(f"Saving to: {local_path}")
    
    try:
        urlretrieve(url, str(local_path), DownloadProgressBar())
        logger.info(f"Download complete: {local_path}")
        return str(local_path)
    except Exception as e:
        # Clean up partial download
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(f"Failed to download checkpoint: {e}")


def clear_cache():
    """Remove all cached checkpoints"""
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Cleared cache: {CACHE_DIR}")
if __name__ == '__main__':
    # Test download
    test_url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
    path = get_checkpoint_path(test_url)
    print(f"Test successful: {path}")
