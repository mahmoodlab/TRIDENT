from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import pyvips

import multiprocessing
from functools import partial

Image.MAX_IMAGE_PIXELS = None

# Bioformats
BIOFORMAT_EXTENSIONS = {
    '.tif', '.tiff', '.ndpi', '.svs', '.lif', '.ims', '.vsi', '.bif', '.btf',
    '.mrxs', '.scn', '.ome.tiff', '.ome.tif', '.h5', '.hdf', '.hdf5', '.he5',
    '.dicom', '.dcm', '.ome.xml', '.zvi', '.pcoraw', '.jp2', '.qptiff', '.nrrd', '.ome.btf', '.fg7'
}

# PIL
PIL_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# OpenSlide
OPENSLIDE_EXTENSIONS = {'.svs', '.tif', '.dcm', '.vms', '.vmu', '.ndpi', '.scn', '.mrxs', '.tiff', '.svslide', '.bif', '.czi'}

# Combined with CZI 
SUPPORTED_EXTENSIONS = BIOFORMAT_EXTENSIONS | PIL_EXTENSIONS | {'.czi'}

class AnyToTiffConverter:
    """
    A class to convert images to TIFF format with options for resizing and pyramidal tiling.
    
    Attributes:
        job_dir (str): Directory to save converted images.
        bigtiff (bool): Flag to enable the creation of BigTIFF files.
    """
    def __init__(self, job_dir: str, bigtiff: bool = False):
        """
        Initializes the Converter with a job directory and BigTIFF support.

        Args:
            job_dir (str): The directory where converted images will be saved.
            bigtiff (bool): Enable or disable BigTIFF file creation.
        """
        self.job_dir = job_dir
        self.bigtiff = bigtiff
        os.makedirs(job_dir, exist_ok=True)

    def process_file(self, input_file: str, mpp: float, zoom: float) -> None:
        """
        Process a single image file to convert it into TIFF format.

        Args:
            input_file (str): Path to the input image file.
            mpp (float): Microns per pixel value for the output image.
            zoom (float): Zoom factor for image resizing, e.g., 0.5 is reducing the image by a factor.
        """
        worker_id = multiprocessing.current_process().name
        try:
            tqdm.write(f"[{worker_id}] Processing: {os.path.basename(input_file)}")
            img_name = os.path.splitext(os.path.basename(input_file))[0]
            save_path = os.path.join(self.job_dir, f"{img_name}.tiff")

            # Step 1: Open the source file with pyvips in streaming mode.
            img = pyvips.Image.new_from_file(input_file, access="sequential")

            # Step 2: Resize if necessary (this is also a streaming operation).
            if zoom != 1.0:
                img = img.resize(zoom)

            new_mpp = mpp / zoom

            # Step 3: Save the pyvips image directly to a pyramidal TIFF.
            self._save_tiff(img, save_path, new_mpp)
            tqdm.write(f"[{worker_id}] Finished: {os.path.basename(save_path)}")
            
        except Exception as e:
            tqdm.write(f"[{worker_id}] Error processing {input_file} with pyvips: {e}")
            # Fallback for formats pyvips might not support, like CZI
            if input_file.lower().endswith('.czi'):
                tqdm.write(f"[{worker_id}] Attempting fallback for CZI...")
                try:
                    # The fallback is memory-intensive and may fail for large files
                    numpy_img = self._read_czi_image(input_file, zoom)
                    new_mpp = mpp / zoom
                    pyvips_img = pyvips.Image.new_from_array(numpy_img)
                    self._save_tiff(pyvips_img, save_path, new_mpp)
                    tqdm.write(f"[{worker_id}] CZI fallback successful.")
                except Exception as fallback_e:
                    tqdm.write(f"[{worker_id}] CZI fallback also failed: {fallback_e}")

    def _read_czi_image(self, file_path: str, zoom: float = 1) -> np.ndarray:
        """
        Fallback function specifically for reading CZI files into a NumPy array.

        Args:
            file_path (str): Path to the CZI image file.
            zoom (float): Zoom factor for resizing.

        Returns:
            np.ndarray: A NumPy array representing the image.
        """
        try:
            import pylibCZIrw.czi as pyczi
        except ImportError:
            raise ImportError("pylibCZIrw is required for CZI files.")
        with pyczi.open_czi(file_path) as czidoc:
            # Using read_mosaic for robustness with CZI files
            mosaic_data, _ = czidoc.read_mosaic(C=0, zoom=zoom)
            return mosaic_data

    def _get_mpp(self, mpp_data: pd.DataFrame, input_file: str) -> float:
        """
        Retrieve the MPP (Microns per Pixel) value for a specific file from a DataFrame.

        Args:
            mpp_data (pd.DataFrame): DataFrame containing MPP values.
            input_file (str): Filename to search for in the DataFrame.

        Returns:
            float: MPP value for the file.
        """
        filename = os.path.basename(input_file)
        mpp_row = mpp_data.loc[mpp_data['wsi'] == filename, 'mpp']
        if mpp_row.empty:
            raise ValueError(f"No MPP found for {filename} in CSV.")
        return float(mpp_row.values[0])

    def _save_tiff(self, pyvips_img: pyvips.Image, save_path: str, mpp: float) -> None:
        """
        Save a pyvips image object as a pyramidal TIFF image.

        Args:
            pyvips_img (pyvips.Image): The pyvips image object to save.
            save_path (str): The full path where the TIFF file will be saved.
            mpp (float): Microns per pixel value of the output TIFF image.
        """
        pyvips_img.tiffsave(
            save_path,
            bigtiff=self.bigtiff,
            pyramid=True,
            tile=True,
            tile_width=256,
            tile_height=256,
            compression='jpeg',
            resunit=pyvips.enums.ForeignTiffResunit.CM,
            xres=1. / (mpp * 1e-4),
            yres=1. / (mpp * 1e-4)
        )

    def process_all(self, input_dir: str, mpp_csv: str, downscale_by: int = 1, num_workers: int = 0) -> None:
        """
        Process all eligible image files in a directory using multiple processes.

        Args:
            input_dir (str): Directory containing image files to process.
            mpp_csv (str): Path to a CSV file with 2 field: "wsi" with fnames with extensions and "mpp" with the micron per pixel values.
            downscale_by (int): Factor to downscale images by, e.g., to save a 40x image into a 20x one, set downscale_by to 2. 
            num_workers (int): Number of parallel processes to use. If 0, uses all available CPU cores.
        """
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(tuple(SUPPORTED_EXTENSIONS))]
        mpp_df = pd.read_csv(mpp_csv)
        
        tasks = []
        for filename in files:
            try:
                img_path = os.path.join(input_dir, filename)
                mpp = self._get_mpp(mpp_df, img_path)
                tasks.append({'input_file': img_path, 'mpp': mpp, 'zoom': 1.0/downscale_by})
            except ValueError as e:
                print(e)
                continue

        if num_workers <= 0:
            num_workers = multiprocessing.cpu_count()
        print(f"Using {num_workers} worker processes for parallel conversion...")
        
        worker_tasks = [(task['input_file'], task['mpp'], task['zoom']) for task in tasks]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            list(tqdm(pool.starmap(AnyToTiffConverter._process_file_static_wrapper, 
                                  [(self, *task) for task in worker_tasks]), 
                      total=len(tasks), desc="Overall Progress"))
        
    @staticmethod
    def _process_file_static_wrapper(instance, input_file, mpp, zoom):
        """
        A static wrapper to allow calling an instance method in a multiprocessing pool.

        Args:
            instance (AnyToTiffConverter): The instance of the class.
            input_file (str): Path to the input image file.
            mpp (float): Original microns per pixel value.
            zoom (float): Zoom factor for resizing.
        """
        instance.process_file(input_file, mpp, zoom)


if __name__ == "__main__":
    # Example usage.
    converter = AnyToTiffConverter(job_dir='./pyramidal_tiff', bigtiff=True)

    # Convert all images using multiple processes. Set num_workers=0 to use all available cores.
    converter.process_all(input_dir='../wsis/', mpp_csv='../wsis/to_process.csv', downscale_by=1, num_workers=0)

    # Example of to_process.csv specifying the mpp of all WSIs in the dir "../wsis"
    # wsi,mpp
    # 3756144.svs,0.25
    # 4290019.svs,0.25
    # 619709.svs,0.25