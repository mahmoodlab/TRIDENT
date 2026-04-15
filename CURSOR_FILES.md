# Cursor Generated Files

This document tracks files generated using Cursor AI assistant.

## process_tma_masks.py

**Date Created:** 2024-12-16  
**Last Updated:** 2024-12-16

### What it does
Processes TMA (Tissue Microarray) geojson segmentation files and creates block-level visualizations. The script:
1. Combines all `TMA_Map_block*.csv` files into a master CSV file (`TMA_Map_Master.csv`)
2. Maps geojson files to TMA blocks using case IDs extracted from filenames
3. Generates segmentation mask visualizations for each block showing TMA cores with alpha=0.5 black overlay on non-TMA regions

### Input Files
- **TMA Map CSV files:** `/data3/hancock/TMA_Maps/TMA_Map_block*.csv`
  - Format: CSV files with columns `core` (TMA index) and `Case ID` (case_id)
  - Function: `combine_tma_map_csvs()` reads and combines these files
  
- **GeoJSON files:** `/data/hancock_qupath/{folder}/geojson_tma_cores/*.geojson`
  - Format: GeoJSON files named `{case_id}.geojson` containing polygon geometries for TMA cores
  - Function: `find_geojson_files()` discovers these files, `load_geojson()` loads them

### Output Files
- **TMA_Map_Master.csv:** `/data3/hancock/TMA_Visualizations/TMA_Map_Master.csv`
  - Columns: `tma_index`, `case_id`, `block`
  - Function: `combine_tma_map_csvs()` creates this file
  
- **Segmentation visualizations:** `/data3/hancock/TMA_Visualizations/block_{block}/{TMA_type}_segmentation.png`
  - Format: PNG images showing TMA cores (white with black edges) and non-TMA regions (black overlay with alpha=0.5)
  - Organization: Each block has its own subfolder, with separate visualizations for each TMA type (TMA_CD8, TMA_CD3, TMA_HE, etc.)
  - Function: `create_segmentation_visualization()` generates these images
  - Example: `block_1/TMA_CD8_segmentation.png`, `block_1/TMA_CD3_segmentation.png`

### Dependencies
- pandas
- geopandas
- matplotlib
- numpy
- shapely

### Usage
```bash
python3 process_tma_masks.py
```

The script will automatically:
1. Process all CSV files in `/data3/hancock/TMA_Maps/`
2. Find all geojson files in `/data/hancock_qupath/*/geojson_tma_cores/`
3. Create the master CSV and block visualizations in `/data3/hancock/TMA_Visualizations/`

## files/consolidate_features.py

**Date Created:** 2024-12-18  
**Last Updated:** 2024-12-18

### What it does
Consolidates features and patches from block-specific directories to marker-level directories by removing block numbers from the directory structure and file names. The script:
1. Finds all marker directories in the base directory
2. For each marker, finds all `block*_case*` directories
3. Extracts case numbers from directory names (e.g., `block1_case334` -> `case334`)
4. Copies files from `block*_case*/10x_896px_0px_overlap/features_conch_v15/` and `patches/` to consolidated marker-level directories
5. Renames files to replace `block{N}` with `case{N}` in filenames (e.g., `TumorCenter_CD163_block1.h5` -> `TumorCenter_CD163_case334.h5`)

### Input Files
- **Source directories:** `/data3/hancock/trident_feats/{MARKER}/block*_case*/10x_896px_0px_overlap/features_conch_v15/`
  - Format: Feature files (typically `.h5` files) in block-specific directories
  - Function: `consolidate_marker()` processes these directories
  
- **Source directories:** `/data3/hancock/trident_feats/{MARKER}/block*_case*/10x_896px_0px_overlap/patches/`
  - Format: Patch files (typically `.h5` files) in block-specific directories
  - Function: `consolidate_marker()` processes these directories

### Output Files
- **Consolidated features:** `/data3/hancock/trident_feats/{MARKER}/10x_896px_0px_overlap/features_conch_v15/`
  - Format: Feature files renamed to include case number instead of block number
  - Function: `consolidate_marker()` creates this directory and copies/renames files
  - Example: `TumorCenter_CD163_case334.h5` (renamed from `TumorCenter_CD163_block1.h5`)
  
- **Consolidated patches:** `/data3/hancock/trident_feats/{MARKER}/10x_896px_0px_overlap/patches/`
  - Format: Patch files renamed to include case number instead of block number
  - Function: `consolidate_marker()` creates this directory and copies/renames files
  - Example: `TumorCenter_CD163_case334_patches.h5` (renamed from `TumorCenter_CD163_block1_patches.h5`)

### Dependencies
- Standard library: os, re, shutil, pathlib, typing, argparse

### Usage
```bash
# Dry run to see what would be done (recommended first)
python3 files/consolidate_features.py --dry_run

# Process all markers
python3 files/consolidate_features.py

# Process a specific marker
python3 files/consolidate_features.py --marker CD163

# Use a different base directory
python3 files/consolidate_features.py --base_dir /path/to/trident_feats
```

The script will automatically:
1. Find all marker directories in the base directory
2. For each marker, find all `block*_case*` directories
3. Extract case numbers and rename files accordingly
4. Copy files to consolidated marker-level directories
5. Skip files if target already exists (with warning)

