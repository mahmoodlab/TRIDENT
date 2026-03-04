from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from trident.IO import splitext

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

def _process_file_worker(args) -> None:
    """Top-level worker for multiprocessing compatibility."""
    job_dir, bigtiff, input_file, mpp, zoom = args
    converter = AnyToTiffConverter(job_dir=job_dir, bigtiff=bigtiff)
    converter.process_file(input_file=input_file, mpp=mpp, zoom=zoom)



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
        self.detected_mpp = {}
        os.makedirs(job_dir, exist_ok=True)

    def process_file(self, input_file: str, mpp: float, zoom: float) -> None:
        """
        Process a single image file to convert it into TIFF format.

        Args:
            input_file (str): Path to the input image file.
            mpp (float): Microns per pixel value for the output image.
            zoom (float): Zoom factor for image resizing, e.g., 0.5 is reducing the image by a factor.
        """
        try:
            embedded_mpp = self._detect_embedded_mpp(input_file)
            if embedded_mpp is not None:
                self.detected_mpp[input_file] = embedded_mpp
                if abs(embedded_mpp - mpp) > 1e-3:
                    print(
                        f"[Converter] MPP mismatch for {os.path.basename(input_file)}: "
                        f"CSV mpp={mpp:.6f}, embedded mpp={embedded_mpp:.6f}. "
                        "Using CSV value."
                    )

            img_name = splitext(os.path.basename(input_file))[0]
            output_mpp = mpp * (1 / zoom)
            save_path = os.path.join(self.job_dir, f"{img_name}.tiff")

            # Fast path: stream directly from source to pyramidal TIFF when pyvips supports it.
            if self._try_pyvips_convert(input_file=input_file, save_path=save_path, zoom=zoom, mpp=output_mpp):
                return

            # Fallback path: load image via existing readers, then save with pyvips.
            img = self._read_image(input_file, zoom)
            self._save_tiff(img, img_name, output_mpp)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    def _detect_embedded_mpp(self, file_path: str):
        """Try to detect embedded MPP metadata using converter-native readers only."""
        if file_path in self.detected_mpp:
            return self.detected_mpp[file_path]

        mpp = self._detect_embedded_mpp_pyvips(file_path)
        if mpp is not None:
            return mpp
        return self._detect_embedded_mpp_aicsimageio(file_path)

    def _detect_embedded_mpp_pyvips(self, file_path: str):
        try:
            import pyvips
        except Exception:
            return None

        try:
            img = pyvips.Image.new_from_file(file_path, access="sequential")
            if img.get_typeof("xres") == 0:
                return None
            xres = float(img.get("xres"))
            if xres <= 0:
                return None
            # pyvips uses pixels/mm for xres.
            return 1000.0 / xres
        except Exception:
            return None

    def _detect_embedded_mpp_aicsimageio(self, file_path: str):
        if not file_path.lower().endswith(tuple(BIOFORMAT_EXTENSIONS)):
            return None
        try:
            from aicsimageio import AICSImage
        except Exception:
            return None

        try:
            img = AICSImage(file_path)
            px_sizes = img.physical_pixel_sizes
            if px_sizes and px_sizes.X is not None:
                return float(px_sizes.X)
        except Exception:
            return None
        return None

    def _try_pyvips_convert(self, input_file: str, save_path: str, zoom: float, mpp: float) -> bool:
        """
        Attempt a streaming conversion with pyvips.

        Returns
        -------
        bool
            True if conversion succeeded, False if caller should fallback.
        """
        # Keep CZI on the dedicated reader path for compatibility.
        if input_file.lower().endswith(".czi"):
            return False

        try:
            import pyvips
        except ImportError:
            return False

        try:
            img = pyvips.Image.new_from_file(input_file, access="sequential")
            if zoom != 1:
                img = img.resize(zoom)
            self._save_pyvips_tiff(img, save_path, mpp, pyvips)
            return True
        except Exception:
            return False

    def _read_image(self, file_path: str, zoom: float = 1) -> np.ndarray:
        """
        Read and resize an image from the given path.

        Args:
            file_path (str): Path to the image file.
            zoom (float): Zoom factor for resizing, e.g., 0.5 is reducing the image by a factor.

        Returns:
            np.ndarray: Array representing the resized image.
        """
        if file_path.endswith('.czi'):
            try:
                import pylibCZIrw.czi as pyczi
            except ImportError:
                raise ImportError("pylibCZIrw is required for CZI files. Install it with pip install pylibCZIrw.")
            with pyczi.open_czi(file_path) as czidoc:
                return czidoc.read(zoom=zoom)
        if file_path.lower().endswith(tuple(BIOFORMAT_EXTENSIONS)):
            try:
                from aicsimageio import AICSImage
            except ImportError:
                raise ImportError("Install aicsimageio with `pip install aicsimageio` to read this format.")

            img_obj = AICSImage(file_path)
            # Extract first timepoint and first z-plane with channel-aware handling.
            czyx = img_obj.get_image_data("CZYX", T=0)
            if czyx.ndim != 4:
                raise ValueError(f"Unexpected image shape from AICSImage: {czyx.shape}")
            first_z = czyx[:, 0, :, :]  # (C, Y, X)
            if first_z.shape[0] == 1:
                data = first_z[0]
            else:
                data = np.transpose(first_z[:3], (1, 2, 0))  # (Y, X, 3)
            if zoom != 1:
                pil_img = Image.fromarray(data)
                new_size = (int(pil_img.width * zoom), int(pil_img.height * zoom))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                data = np.array(pil_img)

            px_sizes = img_obj.physical_pixel_sizes
            if px_sizes and px_sizes.X is not None:
                self.detected_mpp[file_path] = float(px_sizes.X)
            return data
        else:
            with Image.open(file_path) as img:
                new_size = (int(img.width * zoom), int(img.height * zoom))
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                return np.array(img_resized)

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
            raise ValueError(
                f"No MPP found for {filename} in CSV. "
                "MPP must be provided in the CSV file with columns `wsi,mpp`."
            )
        return float(mpp_row.values[0])

    def _save_tiff(self, img: np.ndarray, img_name: str, mpp: float) -> None:
        """
        Save an image as a pyramidal TIFF image.

        Args:
            img (np.ndarray): Image data to save as a numpy array.
            img_name (str): Image name (without extensions). 
            mpp (float): Microns per pixel value of the output TIFF image.
        """
        save_path = os.path.join(self.job_dir, f"{img_name}.tiff")
        try:
            import pyvips
        except ImportError:
            raise ImportError("pyvips is required for saving pyramidal TIFFs. Install it with pip install pyvips.")
        pyvips_img = pyvips.Image.new_from_array(img)
        self._save_pyvips_tiff(pyvips_img, save_path, mpp, pyvips)

    def _save_pyvips_tiff(self, pyvips_img, save_path: str, mpp: float, pyvips_module) -> None:
        """Save a pyvips image object as pyramidal TIFF."""
        pyvips_img.tiffsave(
            save_path,
            bigtiff=self.bigtiff,
            pyramid=True,
            tile=True,
            tile_width=256,
            tile_height=256,
            compression='jpeg',
            resunit=pyvips_module.enums.ForeignTiffResunit.CM,
            xres=1. / (mpp * 1e-4),
            yres=1. / (mpp * 1e-4)
        )

    def process_all(self, input_dir: str, mpp_csv: str, downscale_by: int = 1, num_workers: int = 1) -> None:
        """
        Process all eligible image files in a directory to convert them to pyramidal TIFF.

        Args:
            input_dir (str): Directory containing image files to process.
            mpp_csv (str): Path to a CSV file with 2 field: "wsi" with fnames with extensions and "mpp" with the micron per pixel values.
            downscale_by (int): Factor to downscale images by, e.g., to save a 40x image into a 20x one, set downscale_by to 2. 
            num_workers (int): Number of parallel workers. Use 1 for sequential mode.
        """
        if downscale_by < 1:
            raise ValueError(f"downscale_by must be >= 1, got {downscale_by}.")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
        if not os.path.isfile(mpp_csv):
            raise ValueError(
                f"MPP CSV not found: {mpp_csv}. "
                "Provide a CSV with columns `wsi,mpp`."
            )

        mpp_df = pd.read_csv(mpp_csv)
        required_cols = {"wsi", "mpp"}
        missing_cols = required_cols - set(mpp_df.columns)
        if missing_cols:
            raise ValueError(
                f"MPP CSV must contain columns {sorted(required_cols)}. "
                f"Missing: {sorted(missing_cols)}"
            )
        if mpp_df.empty:
            raise ValueError("MPP CSV is empty. Provide at least one row with columns `wsi,mpp`.")

        mpp_df = mpp_df.dropna(subset=["wsi", "mpp"]).copy()
        mpp_df["wsi"] = mpp_df["wsi"].astype(str)

        tasks = []
        skipped_missing = []
        skipped_unsupported = []
        for filename in mpp_df["wsi"].tolist():
            img_path = os.path.join(input_dir, filename)
            if not os.path.exists(img_path):
                skipped_missing.append(filename)
                continue
            if not filename.lower().endswith(tuple(SUPPORTED_EXTENSIONS)):
                skipped_unsupported.append(filename)
                continue
            mpp = self._get_mpp(mpp_df, img_path)
            tasks.append((self.job_dir, self.bigtiff, img_path, mpp, 1 / downscale_by))

        if skipped_missing:
            print(f"[Converter] Skipping {len(skipped_missing)} files not found in input_dir.")
        if skipped_unsupported:
            print(f"[Converter] Skipping {len(skipped_unsupported)} files with unsupported extension.")
        if not tasks:
            raise ValueError("No valid conversion tasks found from CSV entries.")

        if num_workers == 0:
            num_workers = mp.cpu_count()

        if num_workers <= 1:
            for _, _, img_path, mpp, zoom in tqdm(tasks, desc="Processing images"):
                self.process_file(img_path, mpp, zoom=zoom)
        else:
            with mp.Pool(processes=num_workers) as pool:
                for _ in tqdm(pool.imap_unordered(_process_file_worker, tasks), total=len(tasks), desc="Processing images"):
                    pass

        # No JVM cleanup required after removing valis_hest from converter path.


if __name__ == "__main__":

    # Example usage. Still experimental. Coverage could be improved.
    converter = AnyToTiffConverter(job_dir='./pyramidal_tiff', bigtiff=False)

    # Convert all images in the dir "../pngs" with mpp specified in to_process.csv. TIFF are saved at the original pixel res.
    converter.process_all(input_dir='../wsis/', mpp_csv='../pngs/to_process.csv', downscale_by=1)

    # Example of to_process.csv specifying the mpp of all WSIs in the dir "../wsis"
    # wsi,mpp
    # 3756144.svs,0.25
    # 4290019.svs,0.25
    # 619709.svs,0.258