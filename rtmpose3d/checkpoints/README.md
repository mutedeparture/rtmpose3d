# Model Checkpoints

This folder contains the model checkpoints for RTMPose3D.

## Files

1. **rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth** (95 MB)
   - RTMDet-M person detector
   - Originally from OpenMMLab
   
2. **rtmw3d-l_cock14-0d4ad840_20240422.pth** (221 MB)
   - RTMW3D-L 3D pose estimator
   - Trained on Cocktail14 dataset

## GitHub Release

These checkpoints are too large for git (>100MB each). They should be:

1. **Uploaded to GitHub Releases** as release assets
2. **Download URLs** added to `rtmpose3d/configs/__init__.py`
3. **Auto-downloaded** by the package on first use

## Creating a Release

1. Go to: https://github.com/mutedeparture/rtmpose3d/releases
2. Click "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: `RTMPose3D v1.0.0 - Initial Release`
5. Attach both .pth files
6. Publish release

## Getting URLs

After publishing the release, the download URLs will be:
```
https://github.com/mutedeparture/rtmpose3d/releases/download/v1.0.0/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
https://github.com/mutedeparture/rtmpose3d/releases/download/v1.0.0/rtmw3d-l_cock14-0d4ad840_20240422.pth
```

Update these URLs in `rtmpose3d/configs/__init__.py`.
