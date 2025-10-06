# Model Checkpoints

This folder is a placeholder for model checkpoints. The actual checkpoint files are **NOT** stored in git due to their large size.

## Checkpoint Files

| File | Size | Description |
|------|------|-------------|
| `rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth` | 95 MB | RTMDet-M person detector (OpenMMLab) |
| `rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth` | 221 MB | RTMW3D-L pose estimator (Cocktail14 dataset) |
| `rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth` | 354 MB | RTMW3D-X pose estimator (Cocktail14 dataset) |

**Total: ~669 MB**

## Hosting

Checkpoints are hosted on **HuggingFace Hub**: [rbarac/rtmpose3d](https://huggingface.co/rbarac/rtmpose3d)

### HuggingFace URLs

```
https://huggingface.co/rbarac/rtmpose3d/resolve/main/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
https://huggingface.co/rbarac/rtmpose3d/resolve/main/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
https://huggingface.co/rbarac/rtmpose3d/resolve/main/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth
```

These URLs are configured in `rtmpose3d/configs/__init__.py` and checkpoints are automatically downloaded on first use.

## Upload to HuggingFace

To upload checkpoints (requires HuggingFace authentication):

```bash
# Login (one time)
hf auth login

# Create repo (if needed)
hf repo create rbarac/rtmpose3d --type model

# Upload checkpoints from cache directory
cd ~/.cache/rtmpose3d/checkpoints/
hf upload rbarac/rtmpose3d rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
hf upload rbarac/rtmpose3d rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
hf upload rbarac/rtmpose3d rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth

# Upload model card
cd /home/barac/Projects/mmpose/projects/rtmpose3d
hf upload rbarac/rtmpose3d huggingface_README.md --path-in-repo README.md
```

## Local Cache

After first download, checkpoints are cached at:
```
~/.cache/rtmpose3d/checkpoints/
```

This prevents re-downloading on subsequent uses.
