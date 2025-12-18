# WCS Camera Traps - Download Guide

## Quick Start

```bash
# 1. Install gsutil (if needed)
pip3 install gsutil
export PATH="$PATH:$HOME/Library/Python/3.9/bin"  # Add to ~/.zshrc for permanent

# 2. Download images (resumes automatically if interrupted)
mkdir -p wcs_images
gsutil -m rsync -r gs://public-datasets-lila/wcs-unzipped/animals ./wcs_images/animals

# 3. Build dataset (uses local images, adds to main dataset with annotations)
python3 scripts/build_wcs_camera_traps.py \
  --local-images wcs_images \
  --target-count 7000
```

## Notes

- **Only one folder exists**: `animals/` contains all images (animals, vehicles). Annotations specify category.
- **Person images not available**: Excluded for privacy (11,626 annotations exist, but no image files).
- **Total size**: ~18GB (you only need ~7,000 images, final dataset ~1-2GB).
- **Bounding boxes**: Included in `temp_wcs_annotations.json` (360K animal, 1K vehicle annotations).

## Troubleshooting

**gsutil not found**: Add `export PATH="$PATH:$HOME/Library/Python/3.9/bin"` to `~/.zshrc`

**Nested folder structure**: Script handles `animals/` and `animals/animals/` automatically.
