"""
Consolidate features and patches from block-specific directories to marker-level directories.

This script consolidates features and patches from directories organized by block number
(e.g., block1_case334) into a single marker-level directory, removing the block number
from the directory structure and file names.

Input structure:
    /data3/hancock/trident_feats/{MARKER}/block1_case334/10x_896px_0px_overlap/features_conch_v15/
    /data3/hancock/trident_feats/{MARKER}/block1_case334/10x_896px_0px_overlap/patches/

Output structure:
    /data3/hancock/trident_feats/{MARKER}/10x_896px_0px_overlap/features_conch_v15/
    /data3/hancock/trident_feats/{MARKER}/10x_896px_0px_overlap/patches/

File renaming: block1_case334_{suffix} --> case334_{suffix}
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple
import argparse


def extract_case_info(block_dir_name: str) -> Tuple[str, str]:
    """
    Extract case number and optional suffix from directory name.
    
    Examples:
        block1_case334 -> ('case334', '')
        block1_case334_suffix -> ('case334', '_suffix')
        block2_case123_extra -> ('case123', '_extra')
    
    Args:
        block_dir_name: Directory name like 'block1_case334' or 'block1_case334_suffix'
    
    Returns:
        Tuple of (case_with_number, optional_suffix)
    """
    # Pattern: block{N}_case{N}{optional_suffix}
    pattern = r'block\d+_(case\d+)(.*)'
    match = re.match(pattern, block_dir_name)
    
    if match:
        case_part = match.group(1)  # e.g., 'case334'
        suffix = match.group(2)     # e.g., '' or '_suffix'
        return case_part, suffix
    else:
        # Fallback: try to extract case number directly
        case_match = re.search(r'case\d+', block_dir_name)
        if case_match:
            remaining = block_dir_name[case_match.end():]
            return case_match.group(0), remaining
        return block_dir_name, ''


def rename_file(old_name: str, case_part: str, suffix: str) -> str:
    """
    Rename file by replacing block*_case* pattern with case* pattern.
    
    Args:
        old_name: Original filename (e.g., 'TumorCenter_CD163_block1.h5')
        case_part: Case part (e.g., 'case334')
        suffix: Optional suffix from directory name
    
    Returns:
        New filename (e.g., 'TumorCenter_CD163_case334.h5')
    """
    # First try to replace block*_case* pattern if it exists in filename
    new_name = re.sub(r'block\d+_case\d+', case_part, old_name)
    
    # If no replacement happened, try replacing block{N} with case{N}
    if new_name == old_name:
        # Extract case number from case_part (e.g., 'case334' -> '334')
        case_num_match = re.search(r'case(\d+)', case_part)
        if case_num_match:
            case_num = case_num_match.group(1)
            # Replace block{N} with case{N}
            new_name = re.sub(r'block\d+', f'case{case_num}', new_name)
        else:
            # Fallback: replace block{N} with case_part
            new_name = re.sub(r'block\d+', case_part, new_name)
    
    # If there's a suffix in the directory name, append it before the extension
    if suffix:
        name_parts = os.path.splitext(new_name)
        new_name = f"{name_parts[0]}{suffix}{name_parts[1]}"
    
    return new_name


def consolidate_marker(marker_dir: Path, base_dir: Path, magnification: str = "10x") -> None:
    """
    Consolidate all block*_case* directories for a single marker.
    
    Args:
        marker_dir: Path to marker directory (e.g., /data3/hancock/trident_feats/CD163)
        base_dir: Base directory for trident_feats
        magnification: Magnification string (e.g., "10x" or "20x")
    """
    marker_name = marker_dir.name
    print(f"\nProcessing marker: {marker_name} ({magnification})")
    
    # Find all block*_case* directories
    block_dirs = [d for d in marker_dir.iterdir() 
                  if d.is_dir() and re.match(r'block\d+_case\d+', d.name)]
    
    if not block_dirs:
        print(f"  No block*_case* directories found for {marker_name}")
        return
    
    print(f"  Found {len(block_dirs)} block directories")
    
    # Build directory name based on magnification
    overlap_dir = f"{magnification}_896px_0px_overlap"
    
    # Process each block directory
    for block_dir in block_dirs:
        case_part, suffix = extract_case_info(block_dir.name)
        print(f"  Processing {block_dir.name} -> {case_part}{suffix}")
        
        # Paths for source and target
        source_features_dir = block_dir / overlap_dir / "features_conch_v15"
        source_patches_dir = block_dir / overlap_dir / "patches"
        
        target_features_dir = marker_dir / overlap_dir / "features_conch_v15"
        target_patches_dir = marker_dir / overlap_dir / "patches"
        
        # Create target directories if they don't exist
        target_features_dir.mkdir(parents=True, exist_ok=True)
        target_patches_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and rename feature files
        if source_features_dir.exists():
            for feature_file in source_features_dir.iterdir():
                if feature_file.is_file():
                    new_name = rename_file(feature_file.name, case_part, suffix)
                    target_path = target_features_dir / new_name
                    
                    if target_path.exists():
                        print(f"    WARNING: Target exists, skipping: {target_path.name}")
                    else:
                        shutil.copy2(feature_file, target_path)
                        print(f"    Copied: {feature_file.name} -> {new_name}")
        
        # Copy and rename patch files
        if source_patches_dir.exists():
            for patch_file in source_patches_dir.iterdir():
                if patch_file.is_file():
                    new_name = rename_file(patch_file.name, case_part, suffix)
                    target_path = target_patches_dir / new_name
                    
                    if target_path.exists():
                        print(f"    WARNING: Target exists, skipping: {target_path.name}")
                    else:
                        shutil.copy2(patch_file, target_path)
                        print(f"    Copied: {patch_file.name} -> {new_name}")


def main():
    """Main function to consolidate features and patches for all markers."""
    parser = argparse.ArgumentParser(
        description="Consolidate features and patches by removing block numbers from directory structure"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/data3/hancock/trident_feats",
        help="Base directory containing marker directories (default: /data3/hancock/trident_feats)"
    )
    parser.add_argument(
        "--marker",
        type=str,
        default=None,
        help="Process only a specific marker (default: process all markers)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without actually copying files"
    )
    parser.add_argument(
        "--magnification",
        type=str,
        default="10x",
        help="Magnification to process (e.g., '10x' or '20x', default: '10x')"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return
    
    # Find all marker directories
    if args.marker:
        marker_dirs = [base_dir / args.marker]
        if not marker_dirs[0].exists():
            print(f"Error: Marker directory does not exist: {marker_dirs[0]}")
            return
    else:
        marker_dirs = [d for d in base_dir.iterdir() 
                      if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(marker_dirs)} marker directories")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE - No files will be copied ===")
    
    for marker_dir in marker_dirs:
        if args.dry_run:
            # In dry run, just show what would be done
            block_dirs = [d for d in marker_dir.iterdir() 
                         if d.is_dir() and re.match(r'block\d+_case\d+', d.name)]
            overlap_dir = f"{args.magnification}_896px_0px_overlap"
            print(f"\nMarker: {marker_dir.name} ({args.magnification})")
            print(f"  Would process {len(block_dirs)} block directories")
            for block_dir in block_dirs[:5]:  # Show first 5
                case_part, suffix = extract_case_info(block_dir.name)
                print(f"    {block_dir.name} -> {case_part}{suffix}")
            if len(block_dirs) > 5:
                print(f"    ... and {len(block_dirs) - 5} more")
        else:
            consolidate_marker(marker_dir, base_dir, args.magnification)
    
    print("\n=== Consolidation complete ===")


if __name__ == "__main__":
    main()

