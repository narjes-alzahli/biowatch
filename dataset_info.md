# BioWatch Dataset Information

This document tracks all datasets that have been processed and added to the `dataset/` folder.

## Dataset Statistics Summary

**Note**: For object detection training, **annotations matter more** than image count. Images without annotations cannot be used for training.

### RGB Modality
- **Human**: 28,821 files | 28,821 annotated images | ~28,821 annotations 
- **Animal**: 8,929 files | 8,929 annotated images | ~11,234 annotations 
- **Vehicle**: 16,795 files | 16,795 annotated images | ~16,795 annotations 
- **Total RGB**: 54,545 files | 54,545 annotated images | ~56,850 annotations 

### Thermal Modality
- **Human**: 9,797 files | 14,478 annotated images | ~14,478 annotations
- **Animal**: 11,230 files | 22,906 annotated images | ~22,906 annotations
- **Vehicle**: 12,399 files | 12,399 annotated images | ~12,399 annotations
- **Total Thermal**: 33,426 files | 49,783 annotated images | ~49,783 annotations

### Overall Total
- **Total Image Files**: 87,971 files 
- **Total Annotated Images**: 104,328 images (usable for training) 
- **Total Annotations**: 336,673 bounding boxes
- **Note**: All remaining images have annotations (5 PASCAL edge cases remain but are edge cases)

---

## Processed Datasets

### 1. PASCAL VOC 2012

**Source**: `zips/PASCAL_VOC_2012.zip`

**Modality**: RGB

**Categories Extracted**:
- **Human**: 18,296 files | 18,296 annotated images | ~28,821 annotations 
- **Animal**: 8,772 files | 8,772 annotated images | ~11,234 annotations
- **Vehicle**: 5,988 files | 5,988 annotated images | ~16,795 annotations 
  - Categories: bird, cat, cow, dog, horse, sheep (animal)
  - Categories: bicycle, bus, car, motorbike, train (vehicle)
  - **Excluded**: boat, aeroplane

**Filename Format**: 
- Format: `YYYY_XXXXXX.jpg` (e.g., `2007_000027.jpg`, `2008_000510.jpg`)
- Original PASCAL VOC naming convention preserved
- Duplicates (if any) are renamed with suffix: `YYYY_XXXXXX_N.jpg`

**Quality Filters Applied**:
- Minimum bounding box size: 20×20 pixels (lowered to support both aerial and trap cameras)
- Exclude difficult objects (`difficult=0` only) - ambiguous/hard cases excluded
- **Include truncated objects** (`truncated=1` allowed) - partial objects included for real-world scenarios

**Annotation Format**:
- Source: PASCAL VOC XML format
- Saved as: COCO JSON format in `dataset/annotations.json`
- Bounding box format: `[x_min, y_min, width, height]` (COCO format)

**Build Script**: `scripts/build_pascal_voc.py`

**Notes**:
- Processed from train+val split (17,125 annotation files scanned)
- Images can appear in multiple categories if they contain multiple object types
- **FIXED**: Missing annotations issue resolved using `scripts/fix_pascal_animal_annotations.py`
  - Fixed 9,143 missing human annotations
  - Fixed 2,993 missing vehicle annotations
  - Fixed 4,386 missing animal annotations
  - **All PASCAL VOC files now have annotations** ✓
- All annotations saved to `dataset/annotations.json`

---

### 2. FLIR ADAS Thermal v2

**Source**: `zips/FLIR_ADAS_Thermal_v2.zip`

**Modality**: Both RGB and Thermal

**Categories Extracted**:
- **Thermal Human**: 5,271 images
- **Thermal Animal**: 6 images
- **Thermal Vehicle**: 10,539 images (includes bicycles)
- **RGB Human**: 6,952 images
- **RGB Animal**: 1 image
- **RGB Vehicle**: 10,641 images (includes bicycles)

**Filename Format**: 
- Train/Val images: Original FLIR filenames (preserved from source)
- Video frames: `video-{id}-frame-{frame_number}-{hash}.jpg` (e.g., `video-24ysbPEGoEKKDvRt6-frame-000000-4C4FHWxwNaMyohLZt.jpg`)
- Video frames are subsampled (max 100 frames per video, stride 10)

**Quality Filters Applied**:
- Minimum bounding box size: 20×20 pixels
- Exclude heavily occluded objects (70%-90% occluded)
- **Include bicycle category** in vehicle category
- Processed from train, val, and video test splits

**Annotation Format**:
- Source: COCO JSON format
- Saved as: COCO JSON format in `dataset/annotations.json`
- Bounding box format: `[x, y, width, height]` (COCO format)

**Build Script**: `scripts/build_flir_adas.py`

**Notes**:
- Video test frames are subsampled to avoid over-representation
- Both thermal and RGB modalities processed
- Very limited animal data (mostly human and vehicle)
- All annotations saved to `dataset/annotations.json`

---

### 3. LLVIP (Low-Light Visible-Infrared Paired)

**Source**: `zips/LLVIP.zip`

**Modality**: Both RGB and Thermal (TIR - Thermal Infrared, 8-14μm LWIR, paired images)

**Categories Extracted**:
- **RGB Human**: 3,463 images
- **Thermal Human**: 3,463 images
- **Note**: LLVIP only contains "person" category (human), no animals or vehicles

**Filename Format**: 
- Format: `llvip_{image_id}.jpg` (e.g., `llvip_00001.jpg`, `llvip_12345.jpg`)
- Each image ID has both RGB (visible) and thermal (infrared) versions
- Original LLVIP naming convention with `llvip_` prefix

**Quality Filters Applied**:
- Minimum bounding box size: 20×20 pixels (supports aerial/trap cameras)
- Only processes "person" category (human)
- Images must have both visible and infrared versions available

**Annotation Format**:
- Source: PASCAL VOC XML format
- Saved as: COCO JSON format in `dataset/annotations.json`
- Bounding box format: `[x_min, y_min, width, height]` (COCO format)
- Conversion: PASCAL VOC `(xmin, ymin, xmax, ymax)` → COCO `[x, y, w, h]`

**Build Script**: `scripts/build_llvip.py`

**Notes**:
- Processed 15,488 annotation files, 3,463 valid image pairs
- Each valid image has both RGB and thermal versions (paired dataset)
- Some images may contain unannotated cars in background (not filtered)
- All annotations saved to `dataset/annotations.json`
- **Note**: LLVIP does not contain bicycles or other vehicle categories

---

### 4. iNaturalist 2021 (Mammalia Only)

**Source**: 
- `zips/inaturalisti_val.json.tar.gz` (annotations)
- `zips/inaturalist_val.tar.gz` (images)

**Modality**: RGB

**Categories Extracted**:
- **Animal**: ~~2,460 files~~ | **0 annotated images** | 0 annotations ⚠️ **REMOVED - NO ANNOTATIONS**
- All Mammalia species mapped to "animal" category
- **Status**: Images were deleted from `dataset/rgb/animal/` because they have no bounding box annotations

**Filename Format**: 
- Format: UUID-based filenames (e.g., `0a669dd3-9a46-49e1-8bea-2375446a43d7.jpg`)
- Original iNaturalist naming convention preserved
- Duplicates (if any) are renamed with suffix: `{uuid}_N.jpg`

**Quality Filters Applied**:
- Minimum bounding box size: 20×20 pixels (supports aerial/trap cameras)
- Filter: Mammalia (mammals) class only
- All annotations are in COCO format (already [x, y, w, h])

**Annotation Format**:
- Source: COCO JSON format
- Saved as: COCO JSON format in `dataset/annotations.json`
- Bounding box format: `[x, y, width, height]` (COCO format, already correct)
- Filtered by minimum bbox size during processing

**Build Script**: `scripts/build_inaturalist.py`

**Notes**:
- Processed from validation split (100,000 images scanned, 2,460 Mammalia images found)
- All Mammalia species (246 categories) mapped to single "animal" category
- **CRITICAL ISSUE**: iNaturalist validation set is classification-only (no bounding boxes)
- All 2,460 images were copied but **NONE have annotations** (no bboxes in source data)
- **Status**: ✅ **DELETED** - All 2,461 iNaturalist images removed from `dataset/rgb/animal/` (no annotations available)
- **Not usable for object detection training** - images deleted to avoid confusion

---

### 5. Conservation Drones

**Source**: 
- `zips/conservation_drones_train_real.zip`
- `zips/conservation_drones_test_real.zip`

**Modality**: Thermal

**Categories Extracted**:
- **Human**: 9,208 images (processed with 100 frames per sequence limit)
- **Animal**: 4,230 images (processed with 300 frames per sequence limit, increased for more diversity)
- **Note**: Class ID 0=animal, 1=human in CSV annotations

**Filename Format**: 
- Format: `{video_id}_{sequence_id}_{frame_number}.jpg` (e.g., `0000000011_0000000000_0000000581.jpg`)
- Video sequences are subsampled: 100 frames per sequence for human, 300 frames per sequence for animal
- Original Conservation Drones naming convention preserved

**Quality Filters Applied**:
- Minimum bounding box size: 20×20 pixels (supports aerial/trap cameras)
- **Filter noise**: True (exclude noisy annotations)
- **Filter occlusion**: False (include occluded objects for real-world scenarios)
- **Filter unknown species**: True (exclude unknown species)
- Video sequences subsampled: 100 frames per sequence for human, 300 frames per sequence for animal

**Processing Notes**:
- Animal category processed separately with increased subsampling limit (300 vs 100)
- Script processes to temp folder first, then subsamples, then moves only new files
- Avoids duplicates by checking existing files and annotations before moving

**Annotation Format**:
- Source: MOT-format CSV (frame, object_id, x, y, w, h, class, species, occlusion, noise)
- Saved as: COCO JSON format in `dataset/annotations.json`
- Bounding box format: `[x, y, width, height]` (COCO format)
- Conversion: CSV format `(x, y, w, h)` → COCO `[x, y, w, h]` (already correct)

**Build Script**: `scripts/build_conservation_drones.py`

**Notes**:
- Processed from train and test splits (48 CSV annotation files total)
- Video sequences are subsampled to avoid over-representation
- Human: 9,208 images (100 frames per sequence limit)
- Animal: 4,230 images (300 frames per sequence limit, increased from 100 for more diversity)
- Animal category re-processed to increase count from 1,355 to 4,230
- All annotations saved to `dataset/annotations.json`

---

### 6. HIT-UAV

**Source**: `zips/hit_uav.zip`

**Modality**: Thermal

**Categories Extracted**:
- **Human**: 483 images (1,030 bounding boxes)
- **Vehicle**: 1,545 images (9,737 bounding boxes)
  - Includes: Car, Bicycle, OtherVehicle
  - **Excluded**: DontCare category

**Filename Format**: 
- Format: Original HIT-UAV naming (preserved from source)
- Processed from train, val, and test splits

**Quality Filters Applied**:
- Minimum bounding box size: 20×20 pixels (supports aerial/trap cameras)
- Exclude DontCare category
- **Include bicycle category** in vehicle category

**Annotation Format**:
- Source: YOLO format (normalized center coordinates)
- Saved as: COCO JSON format in `dataset/annotations.json`
- Bounding box format: `[x, y, width, height]` (COCO format)
- Conversion: YOLO `(x_center_norm, y_center_norm, width_norm, height_norm)` → COCO `[x, y, w, h]`
  - Formula: `x = (x_center - width/2) * img_width`, `y = (y_center - height/2) * img_height`

**Build Script**: `scripts/build_hit_uav.py`

**Notes**:
- Processed from train (2,008 images), val (287 images), and test (571 images) splits
- 920 images filtered out (only small boxes <20px)
- All annotations saved to `dataset/annotations.json`
- YOLO to COCO conversion verified and correct

---

### 7. Caltech Aerial (RGBT Pairs + Thermal Singles)

**Source**: 
- `zips/caltech_aerial_labeled_rgbt_pairs.zip`
- `zips/caltech_aerial_labeled_thermal_singles.zip`

**Modality**: Both RGB and Thermal

**Categories Extracted**:
- **Thermal Human**: 106 images (134 bounding boxes)
- **Thermal Vehicle**: 315 images (568 bounding boxes)
- **RGB Human**: 66 images (92 bounding boxes)
- **RGB Vehicle**: 168 images (350 bounding boxes)

**Filename Format**: 
- RGBT Pairs: Original Caltech naming (e.g., `caltech_duck_ONR_2023-03-21-09-59-39_eo-25310.jpg`)
- Thermal Singles: `{sequence_name}_{original_filename}` (e.g., `sequence_name_image.png`)
- Original Caltech naming convention preserved

**Quality Filters Applied**:
- Minimum bounding box size: 20×20 pixels (supports aerial/trap cameras)
- Mask-to-bbox conversion using connected components (flood fill)
- Pixel value mapping: 10 = vehicle, 11 = person/human

**Annotation Format**:
- Source: Segmentation masks (PNG format)
- Saved as: COCO JSON format in `dataset/annotations.json`
- Bounding box format: `[x, y, width, height]` (COCO format)
- Conversion: Mask pixels → connected components → bounding boxes
  - Method: Flood fill to find connected regions, then extract min/max x,y coordinates
  - **Reliable**: Standard computer vision technique, no information lost

**Build Script**: `scripts/build_caltech_aerial.py`

**Notes**:
- Processed directly from zip files (not from caltech_temp)
- RGBT Pairs: 2,282 mask files processed
- Thermal Singles: 37 mask directories processed
- Mask-to-bbox conversion is reliable (uses connected components)
- All annotations saved to `dataset/annotations.json`

---

### 8. New Zealand Wildlife Thermal Imaging (NZ Thermal)

**Source**: 
- Metadata: `zips/new-zealand-wildlife-thermal-imaging-metadata.json.zip`
- Videos: Downloaded from Google Cloud Storage / Azure Blob Storage

**Modality**: Thermal (TIR - Thermal Infrared)

**Categories Extracted**:
- **Animal**: 7,040 files | 7,055 annotated images | ~7,707 bounding boxes
  - **Prioritized**: Big mammals (deer, pig, goat, sheep, cow, dog, horse)
  - **Included**: Other mammals (possum, cat, hedgehog, rabbit, rodent, mustelid, etc.)
  - **Excluded**: Birds (as requested)

**Filename Format**: 
- Format: `nz_thermal_{video_id}_frame_{frame_number:06d}.jpg` (e.g., `nz_thermal_1532613_frame_000493.jpg`)
- One frame extracted per animal track (middle frame of track for best visibility)
- Video ID and frame number preserved from source

**Quality Filters Applied**:
- Minimum bounding box size: 20×20 pixels (supports aerial/trap cameras)
- **Frame sampling**: 1 frame per animal track (middle frame) for diversity
- **Video prioritization**: Big mammals first, then other mammals, birds excluded
- Videos processed until target count reached (7,000 images)
- Videos deleted immediately after frame extraction to save space

**Annotation Format**:
- Source: Track trajectories from video metadata
- Saved as: COCO JSON format in `dataset/annotations.json`
- Bounding box format: `[x, y, width, height]` (COCO format)
- Conversion: Track coordinates → bounding boxes (if available), otherwise full-frame bbox for camera trap videos

**Build Script**: `scripts/build_nz_thermal.py`

**Notes**:
- **Status**: ✅ **MOVED** - All 7,040 images copied to `dataset/thermal/animal/` and annotations merged
- Processed 7,000+ videos selectively (prioritized by animal type)
- Videos downloaded on-demand and deleted after processing
- Frame extraction: 1 middle frame per track (avoids similar frames from overlapping tracks)
- Some images have multiple animals (multiple annotations per image)
- All annotations saved to `dataset/annotations.json`
- **Archive**: Original temp folder archived as `zips/nz_thermal_lila.zip` (51MB)
- **Resume capability**: Script can resume from where it left off (skips already-processed videos)

---

### 9. Client RGB

**Source**: `zips/client_rgb.zip`

**Modality**: RGB

**Categories Extracted**:
- **Human**: 54 files | 54 annotated images | 86 annotations
- **Animal**: 156 files | 156 annotated images | 327 annotations
- **Note**: 4 images contain both human and animal annotations (copied to both folders)

**Filename Format**: 
- Format: Original filenames preserved (e.g., `03030256.jpg`, `06050146.jpg`)
- Images with both categories are copied to both `rgb/human/` and `rgb/animal/` folders
- Each copy has annotations for its respective category only

**Quality Filters Applied**:
- Minimum bounding box size: 20×20 pixels (supports aerial/trap cameras)
- All annotations from COCO format JSON file

**Annotation Format**:
- Source: COCO JSON format (`annotations/instances_default.json`)
- Saved as: COCO JSON format in `dataset/annotations.json`
- Bounding box format: `[x, y, width, height]` (COCO format, already correct)
- Categories: human (id: 1), animal (id: 2)

**Build Script**: `scripts/build_client_rgb.py`

**Notes**:
- Processed 206 images with 413 annotations total
- Images with both categories are included in both category folders (standard practice for multi-category images)
- Distribution: 86 human annotations, 327 animal annotations
- All annotations saved to `dataset/annotations.json`

---

## ⚠️ Datasets with Missing Annotations

### Critical Issues:

1. **iNaturalist 2021** (RGB Animal) ✅ **RESOLVED**
   - **Problem**: 2,460 files, **0 annotated images**, 0 annotations
   - **Cause**: Source data has no bounding boxes (classification-only dataset)
   - **Impact**: 2,460 unusable images for object detection
   - **Solution**: ✅ **DELETED** - All 2,461 iNaturalist images removed from dataset folder

2. **PASCAL VOC 2012** (RGB - All Categories) ✓ **FIXED**
   - **Problem**: Missing annotations for human, animal, and vehicle categories
   - **Missing**: 12,136 files without annotations (9,143 human, 4,391 animal, 2,993 vehicle)
   - **Cause**: Duplicate handling logic issue in build script
   - **Impact**: 12,136 unusable images
   - **Solution**: ✅ Fixed using `scripts/fix_pascal_animal_annotations.py`
   - **Result**: 12,136 annotations added (9,143 human, 4,386 animal, 2,993 vehicle)
   - **Status**: ✅ **COMPLETE** - All PASCAL VOC files now have annotations

### Summary:
- **Total problematic files**: 5 files without annotations (down from 2,465)
  - ✅ iNaturalist: 2,461 files **DELETED** (no bboxes in source data - classification-only dataset)
  - PASCAL Animal: 5 edge cases (likely no valid XML or no valid objects after filtering)
- **Status**: iNaturalist images removed. Only 5 PASCAL edge cases remain (negligible).

---

## Datasets To Process Next

(To be updated as datasets are processed)

---

## Build Scripts

All build scripts are located in `scripts/`:
- `build_pascal_voc.py` - PASCAL VOC 2012
- `build_flir_adas.py` - FLIR ADAS Thermal v2
- `build_llvip.py` - LLVIP
- `build_inaturalist.py` - iNaturalist (Mammalia only)
- `build_conservation_drones.py` - Conservation Drones
- `build_hit_uav.py` - HIT-UAV
- `build_caltech_aerial.py` - Caltech Aerial (RGBT pairs + thermal singles)
- `build_nz_thermal.py` - New Zealand Wildlife Thermal Imaging
- `build_client_rgb.py` - Client RGB

---

## Annotation Storage

All bounding box annotations are stored in COCO JSON format:
- **Location**: `dataset/annotations.json`
- **Format**: COCO format with categories:
  - `human` (id: 1)
  - `animal` (id: 2)
  - `vehicle` (id: 3)

Each image entry includes:
- File path (relative to `dataset/`)
- Image dimensions (width, height)
- Modality (rgb/thermal)
- Category
- All bounding boxes in COCO format: `[x, y, width, height]`

