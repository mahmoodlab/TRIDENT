# DICOMweb Tutorial

This tutorial demonstrates how to use TRIDENT with whole slide images stored on DICOMweb servers, including Google Cloud Healthcare API.

## 📚 Background

**What is DICOMweb?**

DICOMweb is a RESTful web service standard for accessing medical imaging data over HTTP/HTTPS. It's particularly useful when:
- Your WSIs are stored in DICOM format on a server or cloud storage
- You want to process slides without downloading entire files locally
- You're working with Google Cloud Healthcare API or other DICOM servers
- You need remote access to pathology archives

TRIDENT's `DICOMWebWSI` class implements the WADO-RS (Web Access to DICOM Objects) protocol to stream slide data on-demand.

## 🔧 Setup

### 1. Install Required Dependencies
```bash
pip install requests
pip install google-auth google-auth-httplib2  # Only needed for Google Cloud
```

### 2. Authentication Setup

#### For Google Cloud Healthcare API:

**Option 1: Service Account (Recommended for production)**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**Option 2: User Credentials (For development)**
```bash
gcloud auth application-default login
```

#### For Other DICOMweb Servers:

Set basic authentication credentials:
```bash
export DICOMWEB_USERNAME="your_username"
export DICOMWEB_PASSWORD="your_password"
```

## 📝 URL Format Requirements

**CRITICAL:** DICOMweb URLs must point to a **series** (not study or instance level).

### ✅ Correct Format:
```
https://healthcare.googleapis.com/v1/projects/{project}/locations/{location}/datasets/{dataset}/dicomStores/{dicomStore}/dicomWeb/studies/{studyUID}/series/{seriesUID}
```

### ❌ Common Mistakes:

**Study-level URL (missing series):**
```
https://server/studies/1.2.3.4/  # ❌ Will be rejected
```

**Instance-level URL:**
```
https://server/studies/1.2.3.4/series/5.6.7.8/instances/9.0.1.2  # ⚠️ Will be auto-trimmed
```

TRIDENT will automatically trim instance-level URLs to series level and warn you.

## 🚀 Usage Examples

### Example 1: Process a Single DICOMweb Slide
```python
from trident import load_wsi
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory
import os
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv() 

hf_token = os.environ.get('HF_TOKEN')
login(token=hf_token)

# Load DICOMweb slide
dicomweb_url = "https://healthcare.googleapis.com/v1/projects/my-project/locations/us-central1/datasets/pathology/dicomStores/slides/dicomWeb/studies/1.2.840.113/series/1.3.6.1.4"

slide = load_wsi(
    slide_path=dicomweb_url,
    custom_mpp_keys=None  # DICOMweb extracts MPP automatically from metadata
)

# Standard TRIDENT workflow
print(f"Slide dimensions: {slide.dimensions}")
print(f"Magnification: {slide.mag}x")
print(f"MPP: {slide.mpp:.4f} µm/pixel")

# Segment tissue
segmentation_model = segmentation_model_factory('hest')
slide.segment_tissue(
    segmentation_model=segmentation_model,
    target_mag=segmentation_model.target_mag,
    job_dir='./output'
)

# Extract patches and features
coords_path = slide.extract_tissue_coords(
    target_mag=20,
    patch_size=256,
    save_coords='./output/patches'
)

encoder = encoder_factory('uni_v1')
slide.extract_patch_features(
    patch_encoder=encoder,
    coords_path=coords_path,
    save_features='./output/features',
    device='cuda:0'
)
```

### Example 2: Batch Processing with `run_batch_of_slides.py`

Create a CSV file (`dicomweb_slides.csv`) listing your DICOMweb URLs:
```csv
wsi
https://healthcare.googleapis.com/v1/.../studies/1.2.3/series/4.5.6
https://healthcare.googleapis.com/v1/.../studies/1.2.7/series/8.9.0
https://healthcare.googleapis.com/v1/.../studies/1.2.11/series/12.13.14
```

Then run:
```bash
python run_batch_of_slides.py \
    --task all \
    --wsi_dir /tmp/placeholder \
    --custom_list_of_wsis dicomweb_slides.csv \
    --job_dir ./dicomweb_output \
    --patch_encoder uni_v1 \
    --mag 20 \
    --patch_size 256
```

**Note:** `--wsi_dir` is required but unused when `--custom_list_of_wsis` is provided. Just pass any valid directory path.

### Example 3: Using `run_single_slide.py`
```bash
python run_single_slide.py \
    --slide_path "https://healthcare.googleapis.com/v1/projects/my-project/.../series/1.2.3" \
    --job_dir ./output \
    --patch_encoder uni_v1 \
    --mag 20 \
    --patch_size 256 \
    --gpu 0
```

### Example 4: Reading Specific Regions
```python
from trident import load_wsi
from PIL import Image

slide = load_wsi("https://healthcare.googleapis.com/.../series/1.2.3")

# Read a 512x512 region at level 0 (highest resolution)
region = slide.read_region(
    location=(10000, 15000),  # Top-left corner (x, y) in level 0 coordinates
    level=0,
    size=(512, 512)
)

# Save or process the region
region.save("region.png")

# Get thumbnail
thumbnail = slide.get_thumbnail(size=(1024, 1024))
thumbnail.save("thumbnail.png")
```

### Example 5: Working with Pyramid Levels
```python
slide = load_wsi("https://healthcare.googleapis.com/.../series/1.2.3")

# Inspect pyramid structure
slide.print_pyramid_info()
# Output:
# ======================================================================
# WSI Pyramid Information:
# ======================================================================
#   Full dimensions (level 0): 98304 (w) × 73728 (h)
#   Base MPP: 0.2500 µm/pixel
#   Base magnification: 40.0x
#   Number of levels: 5
#
# Pyramid levels:
#   Level   Width    Height   Downsample   MPP             Mag        Tiles
#   ------- -------- -------- ------------ --------------- ---------- ----------
#   0       98304    73728      1.00×      0.2500 µm/px    40.0×       2352
#   1       49152    36864      2.00×      0.5000 µm/px    20.0×        588
#   2       24576    18432      4.00×      1.0000 µm/px    10.0×        147
#   3       12288     9216      8.00×      2.0000 µm/px     5.0×         37
#   4        6144     4608     16.00×      4.0000 µm/px     2.5×         10

# Get best level for target magnification
level = slide.get_best_level_for_mag(target_mag=20.0)  # Returns level 1

# Get MPP for a specific level
mpp = slide.get_mpp_for_level(level=2)  # Returns 1.0 µm/pixel

# Get magnification for a specific level
mag = slide.get_mag_for_level(level=2)  # Returns 10.0x
```

## 🔍 Advanced Features

### Custom Authentication

For non-Google DICOMweb servers with custom authentication:
```python
import requests
from trident.wsi_objects import DICOMWebWSI

class CustomAuth(requests.auth.AuthBase):
    def __init__(self, api_key):
        self.api_key = api_key
    
    def __call__(self, r):
        r.headers['X-API-Key'] = self.api_key
        return r

slide = DICOMWebWSI(
    slide_path="https://your-dicom-server.com/.../series/1.2.3",
    auth=CustomAuth(api_key="your_api_key")
)
```

### Custom Headers
```python
slide = DICOMWebWSI(
    slide_path="https://your-dicom-server.com/.../series/1.2.3",
    headers={
        'X-Custom-Header': 'value',
        'Accept-Language': 'en-US'
    }
)
```
> Note: Your configuration may be different. This is just meant to be a representative example.

## 🐛 Troubleshooting

### Issue: "Failed to authenticate with Google Cloud"

**Solution:**
```bash
# Check if credentials are set
echo $GOOGLE_APPLICATION_CREDENTIALS

# Or login with user credentials
gcloud auth application-default login

# Verify permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

### Issue: "DICOMweb URL must be at series level"

**Cause:** You provided a study-level URL without specifying a series.

**Solution:** Add the series UID to your URL:
```
# Wrong:
https://server/studies/1.2.3.4/

# Correct:
https://server/studies/1.2.3.4/series/5.6.7.8/
```

### Issue: Frame fetch failures or black tiles

**Symptoms:** Console shows messages like:
```
Frame 42 is black/corrupt (mean=8.3), skipping
Frame 108 unavailable (status 404)
```

**Common Causes:**
1. **Sparse tiling**: Some DICOM encoders create sparse tile grids where edge tiles may not exist
2. **Corrupt frames**: Individual frames may be corrupted during encoding
3. **Coordinate misalignment**: Metadata dimension swapping (automatically detected and fixed)

**What TRIDENT Does:**
- Automatically skips missing or corrupt frames
- Fills gaps with white background
- Validates tile intensity (skips suspiciously dark tiles)
- Logs warnings for missing frames (only first 3 per region)

**If you see many failures:**
```python
# Try reading at a different level
region = slide.read_region((0, 0), level=1, size=(1024, 1024))

# Or use the thumbnail
thumbnail = slide.get_thumbnail(size=(512, 512))
```

### Issue: Slow performance

**Optimization tips:**

1. **Use appropriate pyramid levels:**
```python
# Don't read massive regions at level 0
# Instead, use lower-resolution levels for overview
level = slide.get_best_level_for_mag(target_mag=10.0)  # Use 10x instead of 40x
```

2. **Batch coordinate extraction leverages caching:**
```python
# Frames are automatically cached per-process
# Overlapping regions benefit from frame reuse
```

3. **Process slides in parallel:**
```bash
# Use multiple workers for batch processing
python run_batch_of_slides.py \
    --task all \
    --custom_list_of_wsis dicomweb_slides.csv \
    --max_workers 4 \
    --gpu 0
```

### Issue: "Unable to extract MPP from DICOM metadata"

**Solution:** Some DICOM files don't include pixel spacing. You can provide it manually:
```python
slide = load_wsi(
    slide_path="https://server/.../series/1.2.3",
    mpp=0.25  # Manually specify 0.25 µm/pixel (40x)
)
```

## 📊 Metadata Extraction

DICOMweb automatically extracts rich metadata:
```python
slide = load_wsi("https://healthcare.googleapis.com/.../series/1.2.3")

# Pyramid information
print(f"Dimensions: {slide.dimensions}")
print(f"Number of levels: {slide.level_count}")
print(f"Level dimensions: {slide.level_dimensions}")
print(f"Level downsamples: {slide.level_downsamples}")

# Resolution information
print(f"MPP: {slide.mpp:.4f} µm/pixel")
print(f"Magnification: {slide.mag:.1f}x")

# Access raw DICOM metadata
for level, instance in enumerate(slide.instances):
    print(f"Level {level}:")
    print(f"  SOP Instance UID: {instance['sop_instance_uid']}")
    print(f"  Dimensions: {instance['cols']}x{instance['rows']}")
    print(f"  Tile size: {instance['tile_cols']}x{instance['tile_rows']}")
    print(f"  Number of frames: {instance['num_frames']}")
```

## 🎯 Best Practices

1. **Always use series-level URLs** - Study-level URLs are ambiguous (one study can have multiple series)

2. **Verify authentication before large jobs** - Test with a single slide first:
```bash
   python run_single_slide.py --slide_path "https://..." --job_dir ./test
```

3. **Monitor frame fetch success rate** - Some failures are normal for sparse tiles, but if >50% fail, investigate

4. **Use appropriate magnifications** - DICOMweb streams data on-demand, so lower magnifications are much faster:
   - 40x (level 0): Full resolution, slowest
   - 20x (level 1): Good for most feature extraction
   - 10x (level 2): Fast for initial exploration
   - 5x (level 3): Very fast for thumbnails/overview

5. **Leverage TRIDENT's caching** - Frames are cached per-session, so overlapping patch extraction is efficient

## 📚 Additional Resources

- [DICOMweb Standard](https://www.dicomstandard.org/using/dicomweb)
- [Google Cloud Healthcare API Documentation](https://cloud.google.com/healthcare-api/docs/dicom)
- [TRIDENT Documentation](https://trident-docs.readthedocs.io/en/latest/)

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Open an issue on [GitHub](https://github.com/mahmoodlab/trident/issues)
