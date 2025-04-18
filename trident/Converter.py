from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

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
        try:
            img_name = os.path.splitext(os.path.basename(input_file))[0]
            img = self._read_image(input_file, zoom)
            self._save_tiff(img, img_name, mpp * (1/zoom))
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

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
                from valis_hest.slide_io import BioFormatsSlideReader
            except ImportError:
                raise ImportError("Install valis_hest with `pip install valis_hest` and JVM with `sudo apt-get install maven`.")
            reader = BioFormatsSlideReader(file_path) 
            reader.create_metadata()
            img = reader.slide2image(level=int(1/zoom)-1)  # @TODO: Assumes each level 2x small than the higher one.
            return img
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
            raise ValueError(f"No MPP found for {filename} in CSV.")
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

    def process_all(self, input_dir: str, mpp_csv: str, downscale_by: int = 1) -> None:
        """
        Process all eligible image files in a directory to convert them to pyramidal TIFF.

        Args:
            input_dir (str): Directory containing image files to process.
            mpp_csv (str): Path to a CSV file with 2 field: "wsi" with fnames with extensions and "mpp" with the micron per pixel values.
            downscale_by (int): Factor to downscale images by, e.g., to save a 40x image into a 20x one, set downscale_by to 2. 
        """
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(tuple(SUPPORTED_EXTENSIONS))]
        mpp_df = pd.read_csv(mpp_csv)
        for filename in tqdm(files, desc="Processing images"):
            img_path = os.path.join(input_dir, filename)
            mpp = self._get_mpp(mpp_df, img_path)
            try:
                with Image.open(img_path) as img:
                    size = img.size
            except Exception:
                size = "Unknown"
            tqdm.write(f"Processing {filename} | Size: {size}")
            self.process_file(img_path, mpp, zoom=1/downscale_by)

        #clean up 
        try:
            from valis_hest import slide_io
            slide_io.kill_jvm() 
        except:
            pass


if __name__ == "__main__":

    # Example usage. Still experimental. Coverage could be improved.
    converter = AnyToTiffConverter(job_dir='./pyramidal_tiff', bigtiff=False)

    # Convert all images in the dir "../pngs" with mpp specified in to_process.csv. TIFF are saved at the original pixel res.
    converter.process_all(input_dir='../wsis/', mpp_csv='../pngs/to_process.csv', downscale_by=1)

    # Example of to_process.csv specifying the mpp of all WSIs in the dir "../wsis"
    # wsi,mpp
    # 3756144.svs,0.25
    # 4290019.svs,0.25
    # 619709.svs,0.25