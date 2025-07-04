from __future__ import annotations
import os
import sys
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from inspect import signature
import geopandas as gpd
import pandas as pd 

from trident import load_wsi, WSIReaderType
from trident.IO import create_lock, remove_lock, is_locked, update_log, collect_valid_slides
from trident.Maintenance import deprecated
from trident.Converter import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS


class Processor:

    def __init__(
        self,
        job_dir: str,
        wsi_source: str,
        wsi_ext: List[str] = None,
        wsi_cache: Optional[str] = None,
        clear_cache: bool = False,
        skip_errors: bool = False,
        custom_mpp_keys: Optional[List[str]] = None,
        custom_list_of_wsis: Optional[str] = None,
        max_workers: Optional[int] = None,
        reader_type: Optional[WSIReaderType] = None,
        search_nested: bool = False, 
    ) -> None:
        """
        The `Processor` class handles all preprocessing steps starting from whole-slide images (WSIs). 
    
        Available methods:
            - `run_segmentation_job`: Performs tissue segmentation on all slides managed by the processor.
            - `run_patching_job`: Extracts patch coordinates from the segmented tissue regions of slides.
            - `run_patch_feature_extraction_job`: Extracts patch-level features using a specified patch encoder.
                - Deprecated alias: `run_feature_extraction_job`
            - `run_slide_feature_extraction_job`: Extracts slide-level features using a specified slide encoder.
            
        Parameters:
            job_dir (str): 
                The directory where the results of processing, including segmentations, patches, and extracted features, 
                will be saved. This should be an existing directory with sufficient storage.
            wsi_source (str): 
                The directory containing the WSIs to be processed. This can either be a local directory 
                or a network-mounted drive. All slides in this directory matching the specified file 
                extensions will be considered for processing.
            wsi_ext (List[str]): 
                A list of accepted WSI file extensions, such as ['.ndpi', '.svs']. This allows for 
                filtering slides based on their format. If set to None, a default list of common extensions 
                will be used. Defaults to None.
            wsi_cache (str, optional): 
                [DEPRECATED as of v0.2.0] An optional directory for caching WSIs locally. If specified, slides will be copied 
                from the source directory to this local directory before processing, improving performance 
                when the source is a network drive. Defaults to None.
            clear_cache (str, optional):
                [DEPRECATED as of v0.2.0] A flag indicating whether slides in the cache should be deleted after processing. 
                This helps manage storage space. Defaults to False. 
            skip_errors (bool, optional): 
                A flag specifying whether to continue processing if an error occurs on a slide. 
                If set to False, the process will stop on the first error. Defaults to False.
            custom_mpp_keys (List[str], optional): 
                A list of custom keys in the slide metadata for retrieving the microns per pixel (MPP) value. 
                If not provided, standard keys will be used. Defaults to None.
            custom_list_of_wsis (str, optional): 
                Path to a csv file with a custom list of WSIs to process in a field called 'wsi' (including extensions). If provided, only 
                these slides will be considered for processing. Defaults to None, which means all 
                slides matching the wsi_ext extensions will be processed.
                Note: If `custom_list_of_wsis` is provided, any names that do not match the available slides will be ignored, and a warning will be printed.
            max_workers (int, optional):
                Maximum number of workers for data loading. If None, the default behavior will be used.
                Defaults to None.
            reader_type (WSIReaderType, optional):
                Force the image reader engine to use. Options are are ["openslide", "image", "cucim"]. Defaults to None
                (auto-determine the right engine based on image extension).
            search_nested (bool, optional):  
                If True, the processor will recursively search for WSIs within all subdirectories of `wsi_source`.
                All matching files (based on `wsi_ext`) found at any depth within the directory  
                tree will be included. Each slide will be identified by its relative path to `wsi_source`, but only  
                the filename (excluding directory structure) will be used for downstream outputs (e.g., segmentation filenames).  
                If False, only files directly inside `wsi_source` will be considered.  
                Defaults to False.


        Returns:
            None: This method initializes the class instance and sets up the environment for processing.

        Example
        -------
        Initialize the `Processor` for a directory of WSIs:

        >>> processor = Processor(
        ...     job_dir="results/",
        ...     wsi_source="data/slides/",
        ...     wsi_ext=[".svs", ".ndpi"],
        ... )
        >>> print(f"Processor initialized for {len(processor.wsis)} slides.")

        Raises:
            AssertionError: If `wsi_ext` is not a list or if any extension does not start with a period.
        """
        
        if not (sys.version_info.major >= 3 and sys.version_info.minor >= 9):
            raise EnvironmentError("Trident requires Python 3.9 or above. Python 3.10 is recommended.")

        self.job_dir = job_dir
        self.wsi_source = wsi_source
        self.wsi_ext = wsi_ext or (list(PIL_EXTENSIONS) + list(OPENSLIDE_EXTENSIONS))
        self.skip_errors = skip_errors
        self.custom_mpp_keys = custom_mpp_keys
        self.max_workers = max_workers

        # Validate extensions
        assert isinstance(self.wsi_ext, list), f'wsi_ext must be a list, got {type(self.wsi_ext)}'
        for ext in self.wsi_ext:
            assert ext.startswith('.'), f'Invalid extension: {ext} (must start with a period)'

        # === Collect slide paths and relative paths ===
        full_paths, rel_paths = collect_valid_slides(
            wsi_dir=wsi_source,
            custom_list_path=custom_list_of_wsis,
            wsi_ext=self.wsi_ext,
            search_nested=search_nested,
            max_workers=max_workers,
            return_relative_paths=True
        )

        self.wsi_rel_paths = rel_paths if custom_list_of_wsis else None

        # === Extract mpp column if provided ===
        if custom_list_of_wsis is not None:
            wsi_df = pd.read_csv(custom_list_of_wsis)
            valid_mpps = (
                wsi_df['mpp'].dropna().tolist()
                if 'mpp' in wsi_df.columns else None
            )
        else:
            valid_mpps = None

        print(f'[PROCESSOR] Found {len(full_paths)} valid slides in {wsi_source}.')

        # === Initialize WSIs ===
        self.wsis = []
        for wsi_idx, abs_path in enumerate(full_paths):
            name = os.path.basename(abs_path)
            tissue_seg_path = os.path.join(
                self.job_dir, 'contours_geojson',
                f'{os.path.splitext(name)[0]}.geojson'
            )
            if not os.path.exists(tissue_seg_path):
                tissue_seg_path = None

            slide = load_wsi(
                slide_path=abs_path,
                name=name,
                tissue_seg_path=tissue_seg_path,
                custom_mpp_keys=self.custom_mpp_keys,
                mpp=valid_mpps[wsi_idx] if valid_mpps is not None else None,
                max_workers=self.max_workers,
                reader_type=reader_type,
            )
            self.wsis.append(slide)

    def run_segmentation_job(
        self, 
        segmentation_model: torch.nn.Module, 
        seg_mag: int = 10, 
        holes_are_tissue: bool = False,
        batch_size: int = 16,
        artifact_remover_model: torch.nn.Module = None,
        device: str = 'cuda:0', 
    ) -> str:
        """
        The `run_segmentation_job` function performs tissue segmentation on all slides managed by the processor. 
        It uses a machine learning model to identify tissue regions and saves the resulting segmentations to the 
        output directory. This function is essential for workflows that require detailed tissue delineation.

        Parameters:
            segmentation_model (torch.nn.Module): 
                A pre-trained PyTorch model that performs the tissue segmentation. This model should be compatible 
                with the expected input data format of WSIs.
            seg_mag (int, optional): 
                The magnification level at which segmentation is performed. For example, a value of 10 indicates 
                10x magnification. Defaults to 10.
            holes_are_tissue (bool, optional): 
                Specifies whether to treat holes within tissue regions as part of the tissue. Defaults to False.
            batch_size (int, optional): 
                The batch size for segmentation. Defaults to 16.
            artifact_remover_model (torch.nn.Module, optional): 
                A pre-trained PyTorch model that can remove artifacts from an existing segmentation. Defaults to None.
            device (str): 
                The computation device to use (e.g., 'cuda:0' for GPU or 'cpu' for CPU).

        Returns:
            str: Absolute path to where directory containing contours is saved.

        Example
        -------
        Run a segmentation job with a pre-trained model:

        >>> from segmentation.models import TissueSegmenter
        >>> model = TissueSegmenter()
        >>> processor.run_segmentation_job(segmentation_model=model, seg_mag=20)
        """
        saveto = os.path.join(self.job_dir, 'contours')
        os.makedirs(saveto, exist_ok=True)

        sig = signature(self.run_segmentation_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        self.save_config(
            saveto=os.path.join(self.job_dir, '_config_segmentation.json'),
            local_attrs=local_attrs,
            ignore = ['segmentation_model', 'loop', 'valid_slides', 'wsis']
        )

        self.loop = tqdm(self.wsis, desc='Segmenting tissue', total = len(self.wsis))
        for wsi in self.loop:   
            # Check if contour already exists
            if os.path.exists(os.path.join(saveto, f'{wsi.name}.jpg')) and not is_locked(os.path.join(saveto, f'{wsi.name}.jpg')):
                self.loop.set_postfix_str(f'{wsi.name} already segmented. Skipping...')
                update_log(os.path.join(self.job_dir, '_logs_segmentation.txt'), f'{wsi.name}{wsi.ext}', 'Tissue segmented.')
                continue

            # Check if another process has claimed this slide
            if is_locked(os.path.join(saveto, f'{wsi.name}.jpg')):
                self.loop.set_postfix_str(f'{wsi.name} is locked. Skipping...')
                continue

            try:
                self.loop.set_postfix_str(f'Segmenting {wsi}')
                create_lock(os.path.join(saveto, f'{wsi.name}.jpg'))
                update_log(os.path.join(self.job_dir, '_logs_segmentation.txt'), f'{wsi.name}{wsi.ext}', 'LOCKED. Segmenting tissue...')

                # call a function from WSI object to do the work
                gdf_saveto = wsi.segment_tissue(
                    segmentation_model=segmentation_model,
                    target_mag=seg_mag,
                    holes_are_tissue=holes_are_tissue,
                    job_dir=self.job_dir,
                    batch_size=batch_size,
                    device=device
                )

                # additionally remove artifacts for better segmentation.
                if artifact_remover_model is not None:
                    gdf_saveto = wsi.segment_tissue(
                        segmentation_model=artifact_remover_model,
                        target_mag=artifact_remover_model.target_mag,
                        holes_are_tissue=False,
                        job_dir=self.job_dir
                    )

                remove_lock(os.path.join(saveto, f'{wsi.name}.jpg'))

                gdf = gpd.read_file(gdf_saveto, rows=1)
                if gdf.empty:
                    update_log(os.path.join(self.job_dir, '_logs_segmentation.txt'), f'{wsi.name}{wsi.ext}', 'Segmentation returned empty GeoDataFrame.')
                    self.loop.set_postfix_str(f'Empty GeoDataFrame for {wsi.name}.')
                else:
                    update_log(os.path.join(self.job_dir,  '_logs_segmentation.txt'), f'{wsi.name}{wsi.ext}', 'Tissue segmented.')

            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    remove_lock(os.path.join(saveto, f'{wsi.name}.jpg'))
                if self.skip_errors:
                    update_log(os.path.join(self.job_dir, '_logs_segmentation.txt'), f'{wsi.name}{wsi.ext}', f'ERROR: {e}')
                    continue
                else:
                    raise e
                
        # Return the directory where the contours are saved
        return saveto

    def run_patching_job(
        self, 
        target_magnification: int, 
        patch_size: int, 
        overlap: int = 0, 
        saveto: str | None = None, 
        visualize: bool = True,
        min_tissue_proportion: float = 0.,
    ) -> str:
        """
        The `run_patching_job` function extracts patches from the segmented tissue regions of slides. 
        These patches are saved as coordinates in an h5 file for each slide.

        Parameters:
            target_magnification (int): 
                The magnification level for extracting patches. Higher magnifications result in smaller 
                but more detailed patches.
            patch_size (int): 
                The size of each patch in pixels. This refers to the dimensions of the patch at the target magnification.
            overlap (int, optional): 
                The amount of overlap between adjacent patches, specified in pixels. Defaults to 0.
            saveto (str, optional): 
                The directory where patch data and visualizations will be saved. If not provided, a directory 
                name will be generated automatically. Defaults to None.
            visualize (bool, optional): 
                Whether to generate and save visualizations of the patches. Defaults to True.
            min_tissue_proportion: float, optional 
                Minimum proportion of the patch under tissue to be kept. Defaults to 0. 

        Returns:
            str: Absolute path to directory containing patch coordinates.

        Example
        -------
        Extract patches with a size of 256x256 pixels at 20x magnification:

        >>> processor.run_patching_job(
        ...     target_magnification=20, 
        ...     patch_size=256, 
        ...     overlap=32, 
        ...     saveto="output/patches/"
        ... )
        """
        if saveto is None:
            saveto = f"{target_magnification}x_{patch_size}px_{overlap}px_overlap"

        self.target_magnification = target_magnification

        if visualize:
            save_patch_viz = os.path.join(saveto, 'visualization')
            os.makedirs(os.path.join(self.job_dir, save_patch_viz), exist_ok=True)

        os.makedirs(os.path.join(self.job_dir, saveto, 'patches'), exist_ok=True)

        sig = signature(self.run_patching_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        self.save_config(
            saveto=os.path.join(self.job_dir, saveto, '_config_coords.json'),
            local_attrs=local_attrs,
            ignore = ['segmentation_model', 'loop', 'valid_slides', 'wsis']
        )
        self.loop = tqdm(self.wsis, desc=f'Saving tissue coordinates to {saveto}', total = len(self.wsis))
        for wsi in self.loop:    
        
            # Check if patch coords already exist
            if os.path.exists(os.path.join(self.job_dir, saveto, 'patches', f'{wsi.name}_patches.h5')):
                self.loop.set_postfix_str(f'Patch coords already generated for {wsi.name}. Skipping...')
                update_log(os.path.join(self.job_dir, saveto, '_logs_coords.txt'), f'{wsi.name}{wsi.ext}', 'Coords generated')
                continue
            
            # Check if another process has claimed this slide
            if is_locked(os.path.join(self.job_dir, saveto, 'patches', f'{wsi.name}_patches.h5')):
                self.loop.set_postfix_str(f'{wsi.name} is locked. Skipping...')
                continue

            # Check if segmentation exists
            if wsi.tissue_seg_path is None or not os.path.exists(wsi.tissue_seg_path):
                self.loop.set_postfix_str(f'GeoJSON not found for {wsi.name}. Skipping...')
                update_log(os.path.join(self.job_dir, saveto, '_logs_coords.txt'), f'{wsi.name}{wsi.ext}', 'GeoJSON not found.')
                continue
            
            # Check if GeoJSON is empty
            gdf = gpd.read_file(wsi.tissue_seg_path, rows=1)
            if gdf.empty:
                self.loop.set_postfix_str(f'Empty GeoDataFrame for {wsi.name}. Skipping...')
                update_log(os.path.join(self.job_dir, saveto, '_logs_coords.txt'), f'{wsi.name}{wsi.ext}', 'Empty GeoDataFrame.')
                continue

            try:
                self.loop.set_postfix_str(f'Generating patch coords for {wsi.name}{wsi.ext}')
                update_log(os.path.join(self.job_dir, saveto, '_logs_coords.txt'), f'{wsi.name}{wsi.ext}', 'LOCKED. Generating coords...')
                create_lock(os.path.join(self.job_dir, saveto, 'patches', f'{wsi.name}_patches.h5'))

                # save tissue coords
                wsi.extract_tissue_coords(
                    target_mag=target_magnification,
                    patch_size=patch_size,
                    save_coords=os.path.join(self.job_dir, saveto),
                    overlap=overlap,
                    min_tissue_proportion=min_tissue_proportion,
                )

                # save tissue coords visualization
                if visualize:  
                    wsi.visualize_coords(
                        coords_path=os.path.join(self.job_dir, saveto, 'patches', f'{wsi.name}_patches.h5'),
                        save_patch_viz=os.path.join(self.job_dir, save_patch_viz),
                    )

                remove_lock(os.path.join(self.job_dir, saveto, 'patches', f'{wsi.name}_patches.h5'))
                update_log(os.path.join(self.job_dir, saveto, '_logs_coords.txt'), f'{wsi.name}{wsi.ext}', 'Coords generated')
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    remove_lock(os.path.join(self.job_dir, saveto, 'patches', f'{wsi.name}_patches.h5'))
                if self.skip_errors:
                    update_log(os.path.join(self.job_dir, saveto, '_logs_coords.txt'), f'{wsi.name}{wsi.ext}', f'ERROR: {e}')
                    continue
                else:
                    raise e
        
        # Return the directory where the coordinates are saved
        return os.path.join(self.job_dir, saveto)

    @deprecated
    def run_feature_extraction_job(
        self, 
        coords_dir: str, 
        patch_encoder: torch.nn.Module, 
        device: str, 
        saveas: str = 'h5', 
        batch_limit: int = 512, 
        saveto: str | None = None
    ) -> str:
        self.run_patch_feature_extraction_job(
            coords_dir, 
            patch_encoder, 
            device, 
            saveas, 
            batch_limit, 
            saveto,
        )
        
    def run_patch_feature_extraction_job(
        self, 
        coords_dir: str, 
        patch_encoder: torch.nn.Module, 
        device: str, 
        saveas: str = 'h5', 
        batch_limit: int = 512, 
        saveto: str | None = None
    ) -> str:
        """
        The `run_feature_extraction_job` function computes features from the patches generated during the 
        patching step. These features are extracted using a deep learning model and saved in a specified format. 
        This step is often used in workflows that involve downstream analysis, such as classification or clustering.

        Parameters:
            coords_dir (str): 
                Path to the directory containing patch coordinates, which are used to locate patches for feature extraction.
            patch_encoder (torch.nn.Module): 
                A pre-trained PyTorch model used to compute features from the extracted patches.
            device (str): 
                The computation device to use (e.g., 'cuda:0' for GPU or 'cpu' for CPU).
            saveas (str, optional): 
                The format in which extracted features are saved. Can be 'h5' or 'pt'. Defaults to 'h5'.
            batch_limit (int, optional): 
                The maximum number of patches processed in a single batch. Defaults to 512.
            saveto (str, optional): 
                Directory where the extracted features will be saved. If not provided, a directory name will 
                be generated automatically. Defaults to None.

        Returns:
            str: The absolute path to where the features are saved.

        Example
        -------
        Extract features from patches using a pre-trained encoder:

        >>> from models import PatchEncoder
        >>> encoder = PatchEncoder()
        >>> processor.run_feature_extraction_job(
        ...     coords_dir="output/patch_coords/",
        ...     patch_encoder=encoder,
        ...     device="cuda:0"
        ... )
        """
        if saveto is None:
            saveto = os.path.join(coords_dir, f'features_{patch_encoder.enc_name}')

        os.makedirs(os.path.join(self.job_dir, saveto), exist_ok=True)

        sig = signature(self.run_patch_feature_extraction_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        self.save_config(
            saveto=os.path.join(self.job_dir, coords_dir, f'_config_feats_{patch_encoder.enc_name}.json'),
            local_attrs=local_attrs,
            ignore = ['patch_encoder', 'loop', 'valid_slides', 'wsis']
        )

        log_fp = os.path.join(self.job_dir, coords_dir, f'_logs_feats_{patch_encoder.enc_name}.txt')
        self.loop = tqdm(self.wsis, desc=f'Extracting patch features from coords in {coords_dir}', total = len(self.wsis))
        for wsi in self.loop:    
            wsi_feats_fp = os.path.join(self.job_dir, saveto, f'{wsi.name}.{saveas}')
            # Check if features already exist
            if os.path.exists(wsi_feats_fp) and not is_locked(wsi_feats_fp):
                self.loop.set_postfix_str(f'Features already extracted for {wsi}. Skipping...')
                update_log(log_fp, f'{wsi.name}{wsi.ext}', 'Features extracted.')
                continue

            # Check if coords exist
            coords_path = os.path.join(self.job_dir, coords_dir, 'patches', f'{wsi.name}_patches.h5')
            if not os.path.exists(coords_path):
                self.loop.set_postfix_str(f'Coords not found for {wsi.name}. Skipping...')
                update_log(log_fp, f'{wsi.name}{wsi.ext}', 'Coords not found.')
                continue

            # Check if another process has claimed this slide
            if is_locked(wsi_feats_fp):
                self.loop.set_postfix_str(f'{wsi.name} is locked. Skipping...')
                continue

            try:
                self.loop.set_postfix_str(f'Extracting features from {wsi.name}{wsi.ext}')
                create_lock(wsi_feats_fp)
                update_log(log_fp, f'{wsi.name}{wsi.ext}', 'LOCKED. Extracting features...')

                # under construction
                wsi.extract_patch_features(
                    patch_encoder = patch_encoder,
                    coords_path = coords_path,
                    save_features=os.path.join(self.job_dir, saveto),
                    device=device,
                    saveas=saveas,
                    batch_limit=batch_limit
                )

                remove_lock(wsi_feats_fp)
                update_log(log_fp, f'{wsi.name}{wsi.ext}', 'Features extracted.')
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    remove_lock(wsi_feats_fp)
                if self.skip_errors:
                    update_log(log_fp, f'{wsi.name}{wsi.ext}', f'ERROR: {e}')
                    continue
                else:
                    raise e
        
        # Return the directory where the features are saved
        return os.path.join(self.job_dir, saveto)

    def run_slide_feature_extraction_job(
        self,
        slide_encoder: torch.nn.Module,
        coords_dir: str,
        device: str = 'cuda',
        batch_limit: int = 512, 
        saveas: str = 'h5', 
        saveto: str | None = None
    ) -> None:
        """
        Extract slide-level features from whole-slide images (WSIs) using a specified slide encoder.

        This function generates embeddings for WSIs by first ensuring that patch-level features
        required for the slide encoder are available. If patch features are missing, they are
        extracted using an appropriate patch encoder automatically inferred. The extracted slide features are saved in 
        the specified format and directory.

        Args:
            slide_encoder (torch.nn.Module): The slide encoder model used for generating slide-level
                features from patch-level features.
            coords_dir (str): Directory containing coordinates and features required for processing WSIs.
            device (str, optional): Device to use for computations (e.g., 'cuda', 'cpu'). Defaults to 'cuda'.
            batch_limit (int, optional): Maximum number of features processed in a batch during patch
                feature extraction. Defaults to 512.
            saveas (str, optional): File format to save slide features (e.g., 'h5'). Defaults to 'h5'.
            saveto (str | None, optional): Directory to save extracted slide features. If None, the
                directory is auto-generated based on `coords_dir` and `slide_encoder`. Defaults to None.

        Returns:
            str: The absolute path to where the slide embeddings are saved. 

        Workflow:
            1. Verify the compatibility of the slide encoder and patch features.
            2. Check if patch-level features are already extracted for all WSIs. If not, extract them.
            3. Save the configuration for slide feature extraction to maintain reproducibility.
            4. Process each WSI:
                - Skip if patch features required for the WSI are missing.
                - Extract slide features, ensuring proper synchronization in multiprocessing setups.
            5. Log the progress and errors during processing.

        Notes:
            - Patch features are expected in a specific directory structure under `coords_dir`.
            - Slide features are saved in the format specified by `saveas`.
            - Errors can be optionally skipped based on the `self.skip_errors` attribute.

        Raises:
            Exception: Propagates exceptions unless `self.skip_errors` is set to True.

        """
        from trident.slide_encoder_models.load import slide_to_patch_encoder_name
        
        if slide_encoder.enc_name.startswith('mean-'):
            slide_to_patch_encoder_name[slide_encoder.enc_name] = slide_encoder.enc_name.split('mean-')[1] # e.g. mean-resnet18 -> resnet18

        # Setting I/O
        mustbe_patch_encoder = slide_to_patch_encoder_name[slide_encoder.enc_name]
        patch_features_dir = os.path.join(coords_dir, f'features_{mustbe_patch_encoder}')
        if saveto is None:
            saveto = os.path.join(coords_dir, f'slide_features_{slide_encoder.enc_name}')
        os.makedirs(os.path.join(self.job_dir, saveto), exist_ok=True)

        # Run patch feature extraction if some patch features are missing:
        already_processed = []
        if os.path.isdir(os.path.join(self.job_dir, patch_features_dir)):
            already_processed = [os.path.splitext(x)[0] for x in os.listdir(os.path.join(self.job_dir, patch_features_dir)) if x.endswith(saveas)]
            wsi_names = [slide.name for slide in self.wsis]
            already_processed = [x for x in already_processed if x in wsi_names]
        if len(already_processed) < len(self.wsis):
            print(f"[PROCESSOR] Some patch features haven't been extracted in {len(already_processed)}/{len(self.wsis)} WSIs. Starting extraction.")
            from trident.patch_encoder_models.load import encoder_factory
            patch_encoder = encoder_factory(slide_to_patch_encoder_name[slide_encoder.enc_name])
            self.run_patch_feature_extraction_job(
                coords_dir=coords_dir,
                patch_encoder=patch_encoder,
                device=device,
                saveas='h5',  # must use h5 to run slide extraction later to get coords.
                batch_limit=batch_limit,
            )

        sig = signature(self.run_slide_feature_extraction_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        self.save_config(
            saveto=os.path.join(self.job_dir, coords_dir, f'_config_slide_features_{slide_encoder.enc_name}.json'),
            local_attrs=local_attrs,
            ignore=['loop', 'valid_slides', 'wsis']
        )

        self.loop = tqdm(self.wsis, desc=f'Extracting slide features using {slide_encoder.enc_name}', total=len(self.wsis))
        for wsi in self.loop:
            # Check if slide features already exist
            slide_feature_path = os.path.join(self.job_dir, saveto, f'{wsi.name}.{saveas}')
            if os.path.exists(slide_feature_path) and not is_locked(slide_feature_path):
                self.loop.set_postfix_str(f'Slide features already extracted for {wsi.name}. Skipping...')
                update_log(os.path.join(self.job_dir, coords_dir, f'_logs_slide_features_{slide_encoder.enc_name}.txt'), f'{wsi.name}{wsi.ext}', 'Slide features extracted.')
                continue

            # Check if patch features exist
            patch_features_path = os.path.join(self.job_dir, patch_features_dir, f'{wsi.name}.h5')
            if not os.path.exists(patch_features_path):
                self.loop.set_postfix_str(f'Patch features not found for {wsi.name}. Skipping...')
                update_log(os.path.join(self.job_dir, coords_dir, f'_logs_slide_features_{slide_encoder.enc_name}.txt'), f'{wsi.name}{wsi.ext}', 'Patch features not found.')
                continue

            # Check if another process has claimed this slide
            if is_locked(slide_feature_path):
                self.loop.set_postfix_str(f'{wsi.name} is locked. Skipping...')
                continue

            try:
                self.loop.set_postfix_str(f'Extracting slide features for {wsi.name}{wsi.ext}')
                create_lock(slide_feature_path)
                update_log(os.path.join(self.job_dir, coords_dir, f'_logs_slide_features_{slide_encoder.enc_name}.txt'), f'{wsi.name}{wsi.ext}', 'LOCKED. Extracting slide features...')

                # Call the extract_slide_features method
                wsi.extract_slide_features(
                    patch_features_path=patch_features_path,
                    slide_encoder=slide_encoder,
                    device=device,
                    save_features=os.path.join(self.job_dir, saveto)
                )

                remove_lock(slide_feature_path)
                update_log(os.path.join(self.job_dir, coords_dir, f'_logs_slide_features_{slide_encoder.enc_name}.txt'), f'{wsi.name}{wsi.ext}', 'Slide features extracted.')
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    remove_lock(slide_feature_path)
                if self.skip_errors:
                    update_log(os.path.join(self.job_dir, coords_dir, f'_logs_slide_features_{slide_encoder.enc_name}.txt'), f'{wsi.name}{wsi.ext}', f'ERROR: {e}')
                    continue
                else:
                    raise e
        
        return os.path.join(self.job_dir, saveto)

    def save_config(
        self,
        saveto: str,
        local_attrs: Optional[Dict[str, Any]] = None,
        ignore: List[str] = ['valid_slides']
    ) -> None:
        """
        The `save_config` function saves the current configuration of the `Processor` instance to a JSON file. 
        This configuration includes attributes of the instance as well as optional additional parameters 
        provided via the `local_attrs` argument.

        The function filters out attributes specified in the `ignore` list and ensures that only JSON-serializable 
        attributes are included. This makes it ideal for saving configurations in a structured format that can 
        later be reloaded or inspected for reproducibility.

        Parameters:
            saveto (str): 
                The path to the file where the configuration will be saved. This should include the file extension 
                (e.g., "config.json").
            local_attrs (dict, optional): 
                A dictionary of additional attributes to include in the configuration. This can be used to add 
                method-specific parameters or runtime settings. Defaults to None.
            ignore (list, optional): 
                A list of attribute names to exclude from the configuration. This is useful for omitting large 
                or non-serializable objects. Defaults to ['valid_slides'].

        Returns:
            None: The function saves the configuration to the specified file and does not return any value.

        Example
        -------
        Save the current processor configuration to a file:

        >>> processor.save_config(saveto="output/config.json")
        >>> # Check the saved configuration
        >>> with open("output/config.json", "r") as f:
        ...     config = json.load(f)
        ...     print(config)
        """
        import json
        from trident.IO import JSONsaver

        def serialize_safe(obj):
            try:
                return json.loads(json.dumps(obj))  # Ensure the object is JSON-serializable
            except (TypeError, OverflowError):
                return None

        # Merge instance attributes and local_attrs, filtering ignored and unserializable items
        config = {
            k: serialize_safe(v)
            for attr_dict in [vars(self), local_attrs or {}]
            for k, v in attr_dict.items()
            if k not in ignore and serialize_safe(v) is not None
        }

        # Save the combined configuration to the specified file
        with open(saveto, 'w') as f:
            json.dump(config, f, indent=4, cls=JSONsaver)

    def release(self) -> None:
        """
        Release all resources tied to the WSIs held by this Processor instance.
        Frees memory, closes file handles, and clears GPU memory.
        Should be called after processing is complete to avoid memory leaks.
        """
        if hasattr(self, "wsis"):
            for wsi in self.wsis:
                try:
                    wsi.release()
                except Exception:
                    pass
            self.wsis.clear()

        # Also clear loop references (e.g., tqdm)
        if hasattr(self, "loop"):
            self.loop = None

        # Explicit garbage collection and CUDA cache release
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
