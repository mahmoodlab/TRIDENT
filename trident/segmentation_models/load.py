from typing import Dict, Any, Tuple, Optional
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from abc import ABC, abstractmethod

from trident.IO import get_dir, get_weights_path, has_internet_connection


class BaseSegmentationModel(torch.nn.Module, ABC):

    _has_internet = has_internet_connection()

    def __init__(self, freeze: bool = True, confidence_thresh: float = 0.5, **build_kwargs: Dict[str, Any]):
        """
        Initialize Segmentation model wrapper.

        Parameters:
            freeze (bool, optional):
                If True, the model's parameters are frozen (i.e., not trainable) and the model is set to evaluation
                mode. Defaults to True.
            confidence_thresh (float, optional):
                Threshold for prediction confidence. Predictions below this threshold may be filtered out or ignored.
                Default is 0.5. Set to 0.4 to keep more tissue.
            **build_kwargs (dict):
                Additional keyword arguments passed to the internal `_build` method.

        Attributes:
            model (torch.nn.Module):
                The constructed model.
            eval_transforms (Callable):
                Transformations to apply to input data during inference.
            interpolation (int or None):
                cv2.INTER_* flag the patcher uses to resize tiles to `input_size`. None (default)
                keeps the patcher's default (cv2.INTER_LINEAR); a subclass may set it in `_build`.
        """
        super().__init__()
        self.interpolation = None
        self.model, self.eval_transforms = self._build(**build_kwargs)
        self.confidence_thresh = confidence_thresh

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Can be overwritten if model requires special forward pass.
        """
        z = self.model(image)
        return z
        
    @property
    def name(self) -> Optional[str]:
        """Canonical model id — uniform ``.name`` accessor shared across all model families
        (set by ``segmentation_model_factory``)."""
        return getattr(self, "model_name", None)

    @abstractmethod
    def _build(self, **build_kwargs: Dict[str, Any]) -> Tuple[nn.Module, transforms.Compose]:
        """
        Build the segmentation model and preprocessing transforms.
        """
        pass


# Backward-compatible alias for the pre-rename base class.
SegmentationModel = BaseSegmentationModel


class HESTSegmenter(BaseSegmentationModel):

    def __init__(self, **build_kwargs: Dict[str, Any]):
        """
        HESTSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self) -> Tuple[nn.Module, transforms.Compose]:
        """
        Build and load HESTSegmenter model.

        Returns:
            Tuple[nn.Module, transforms.Compose]: Model and preprocessing transforms.
        """

        from torchvision.models.segmentation import deeplabv3_resnet50

        model_ckpt_name = 'deeplabv3_seg_v4.ckpt'
        weights_path = get_weights_path('seg', 'hest')

        # Check if a path is provided but doesn't exist
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the file was not found.")

        # Initialize base model
        model = deeplabv3_resnet50(weights=None)
        model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1, stride=1)

        if not weights_path:
            if not BaseSegmentationModel._has_internet:
                raise FileNotFoundError(
                    f"Internet connection not available and checkpoint not found locally in model registry at trident/segmentation_models/local_ckpts.json.\n\n"
                    f"To proceed, please manually download {model_ckpt_name} from:\n"
                    f"https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    f"and place it at:\nlocal_ckpts.json"
                )

            # If internet is available, download from HuggingFace
            from huggingface_hub import snapshot_download
            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type='model',
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name]
            )

            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        # Load and clean checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = {
            k.replace('model.', ''): v
            for k, v in checkpoint.get('state_dict', {}).items()
            if 'aux' not in k
        }

        model.load_state_dict(state_dict)

        # Store configuration
        self.input_size = 512
        self.precision = torch.float16
        self.target_mag = 10

        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        return model, eval_transforms
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # input should be of shape (batch_size, C, H, W)
        assert len(image.shape) == 4, f"Input must be 4D image tensor (shape: batch_size, C, H, W), got {image.shape} instead"
        logits = self.model(image)['out']
        softmax_output = F.softmax(logits, dim=1)
        predictions = (softmax_output[:, 1, :, :] > self.confidence_thresh).to(torch.uint8)  # Shape: [bs, 512, 512]
        return predictions
        

class JpegCompressionTransform:
    def __init__(self, quality=80):
        self.quality = quality

    def __call__(self, image):
        import cv2
        import numpy as np
        from PIL import Image
        # Convert PIL Image to NumPy array
        image = np.array(image)

        # Apply JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, image = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Convert back to PIL Image
        return Image.fromarray(image)


class GrandQCArtifactSegmenter(BaseSegmentationModel):

    _class_mapping = {
        1: "Normal Tissue",
        2: "Fold",
        3: "Darkspot & Foreign Object",
        4: "PenMarking",
        5: "Edge & Air Bubble",
        6: "OOF",
        7: "Background"
    }

    def __init__(self, **build_kwargs):
        """
        GrandQCArtifactSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, remove_penmarks_only=False):
        """
        Load the GrandQC artifact removal segmentation model.
        Credit: https://www.nature.com/articles/s41467-024-54769-y
        """

        import segmentation_models_pytorch as smp

        self.remove_penmarks_only = remove_penmarks_only  # ignore all other artifacts than penmakrs.
        model_ckpt_name = 'GrandQC_MPP1_state_dict.pth'
        encoder_name = 'timm-efficientnet-b0'
        weights_path = get_weights_path('seg', 'grandqc_artifact')

        # Verify that user-provided weights_path is valid
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Expected checkpoint at '{weights_path}', but the file was not found."
            )

        # Initialize model
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=8,
            activation=None,
        )

        # Attempt to download if file is missing and not already available
        if not weights_path:
            if not BaseSegmentationModel._has_internet:
                raise FileNotFoundError(
                    f"Internet connection not available and checkpoint not found locally.\n\n"
                    f"To proceed, please manually download {model_ckpt_name} from:\n"
                    f"https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    f"and place it at:\nlocal_ckpts.json"
                )

            from huggingface_hub import snapshot_download
            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type='model',
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name],
            )

            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        # Load checkpoint
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

        # Model config
        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 10
        import cv2
        self.interpolation = cv2.INTER_LANCZOS4

        # Evaluation transforms
        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Custom forward pass.
        """
        logits = self.model.predict(image)
        probs = torch.softmax(logits, dim=1)  
        _, predicted_classes = torch.max(probs, dim=1)  
        if self.remove_penmarks_only:
            predictions = torch.where((predicted_classes == 4) | (predicted_classes == 7), 0, 1)
        else:
            predictions = torch.where(predicted_classes > 1, 0, 1)
        predictions = predictions.to(torch.uint8)

        return predictions


class GrandQCSegmenter(BaseSegmentationModel):
    
    def __init__(self, **build_kwargs):
        """
        GrandQCSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        """
        Load the GrandQC tissue detection segmentation model.
        Credit: https://www.nature.com/articles/s41467-024-54769-y
        """
        import segmentation_models_pytorch as smp

        model_ckpt_name = 'Tissue_Detection_MPP10.pth'
        encoder_name = 'timm-efficientnet-b0'
        weights_path = get_weights_path('seg', 'grandqc') 

        # Verify that user-provided weights_path is valid
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Expected checkpoint at '{weights_path}', but the file was not found."
            )

        # Verify checkpoint path
        if not weights_path:
            if not BaseSegmentationModel._has_internet:
                raise FileNotFoundError(
                    f"Internet connection not available and checkpoint not found locally at '{weights_path}'.\n\n"
                    f"To proceed, please manually download {model_ckpt_name} from:\n"
                    f"https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    f"and place it at:\nlocal_ckpts.json"
                )

            from huggingface_hub import snapshot_download
            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type='model',
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name],
            )
            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        # Initialize model
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=2,
            activation=None,
        )

        # Load checkpoint
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

        # Model config
        self.input_size = 512
        self.precision = torch.float16
        self.target_mag = 1

        # Evaluation transforms
        eval_transforms = transforms.Compose([
            JpegCompressionTransform(quality=80),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Custom forward pass.
        """
        logits = self.model.predict(image)
        probs = torch.softmax(logits, dim=1)  
        max_probs, predicted_classes = torch.max(probs, dim=1)  
        predictions = (max_probs >= self.confidence_thresh) * (1 - predicted_classes)
        predictions = predictions.to(torch.uint8)

        return predictions


class TissueQCSegmenter(BaseSegmentationModel):
    """
    TissueQC: background, tissue and artifact segmentation at 5x in one forward pass.

    A U-Net with an EfficientNet-B0 encoder predicts three classes per pixel at 2 um/px: 0 background
    (glass, area outside the slide, artifacts lying on bare glass), 1 tissue, 2 artifact on tissue
    (fold, ink, out-of-focus area, knife line, bubble, debris). The same prediction gives the default
    tissue mask (artifacts on tissue kept) or, with ``remove_artifacts``, the mask with artifact regions
    cut out, so no separate artifact model is needed. Weights: https://huggingface.co/JaumeLab/TissueQC
    (Apache-2.0), release pinned by ``HF_REVISION``.
    """

    HF_REPO = "JaumeLab/TissueQC"
    HF_REVISION = "v1.1.0"            # weights release on the Hub (tag); v1.0.0 is the first release
    CKPT_NAME = "model.safetensors"
    detects_artifacts = True          # artifacts come from this model's own prediction

    # slide-level mask rules (see `trident.segmentation_models.model_zoo.tissueqc.tissue_masks`)
    mask_mpp = 8.0                    # masks and contours are built on 8 um/px
    min_artifact_width_um = 24.0      # narrower artifact regions (cracks, lines) stay tissue
    min_artifact_area_mm2 = 0.005     # smaller artifact regions (specks) stay tissue
    min_tissue_area_mm2 = 0.01

    def __init__(self, remove_artifacts: bool = False, **build_kwargs: Dict[str, Any]):
        """
        TissueQCSegmenter initialization.

        Parameters:
            remove_artifacts (bool, optional):
                Cut artifact regions out of the tissue mask. Defaults to False.
            **build_kwargs (dict):
                Forwarded to ``BaseSegmentationModel``. ``confidence_thresh`` is the artifact probability
                threshold: a pixel is artifact when p(artifact) >= this value. Lower it to find more artifacts.
        """
        super().__init__(**build_kwargs)
        self.remove_artifacts = remove_artifacts

    def _build(self) -> Tuple[nn.Module, transforms.Compose]:
        """
        Build and load the TissueQC model.

        Returns:
            Tuple[nn.Module, transforms.Compose]: Model and preprocessing transforms.
        """
        import segmentation_models_pytorch as smp
        from safetensors.torch import load_file

        weights_path = get_weights_path('seg', 'tissueqc')

        # Check if a path is provided but doesn't exist
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the file was not found.")

        if not weights_path:
            if not BaseSegmentationModel._has_internet:
                raise FileNotFoundError(
                    f"Internet connection not available and checkpoint not found locally in model registry at trident/segmentation_models/local_ckpts.json.\n\n"
                    f"To proceed, please manually download {self.CKPT_NAME} from:\n"
                    f"https://huggingface.co/{self.HF_REPO}/tree/{self.HF_REVISION}\n"
                    f"and place it at:\nlocal_ckpts.json"
                )

            # If internet is available, download from HuggingFace
            from huggingface_hub import snapshot_download
            checkpoint_dir = snapshot_download(
                repo_id=self.HF_REPO,
                repo_type='model',
                revision=self.HF_REVISION,
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[self.CKPT_NAME]
            )
            weights_path = os.path.join(checkpoint_dir, self.CKPT_NAME)

        model = smp.Unet(encoder_name="tu-efficientnet_b0", encoder_weights=None, classes=3,
                         decoder_channels=(128, 64, 48, 32, 16))
        model.load_state_dict(load_file(weights_path))

        # Store configuration
        self.input_size = 512
        self.precision = torch.bfloat16
        self.target_mag = 5
        # 5x tiles are cheap to infer and costly to decode: small batches spread over many loader workers
        self.batch_size = 4
        self.num_workers = 32

        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Class map per pixel: uint8 tensor of shape (batch_size, H, W) with 0 background, 1 tissue, 2 artifact.
        """
        assert len(image.shape) == 4, f"Input must be 4D image tensor (shape: batch_size, C, H, W), got {image.shape} instead"
        probs = self.model(image).float().softmax(1)
        classes = (probs[:, 0] < 0.5).to(torch.uint8)
        classes[probs[:, 2] >= self.confidence_thresh] = 2
        return classes

    def to_contours(self, class_map, level0_mpp: float, holes_are_tissue: bool = True):
        """
        Tissue contours from the stitched class map of a slide (used by ``WSI.segment_tissue``).

        Parameters:
            class_map (np.ndarray): Stitched output of :meth:`forward` at ``10 / target_mag`` um/px.
            level0_mpp (float): Pixel size of the slide's level 0, used to scale contours to level-0 pixels.
            holes_are_tissue (bool, optional): Fill holes enclosed by tissue. Defaults to True.

        Returns:
            gpd.GeoDataFrame: Tissue polygons (with holes) in level-0 pixels.
        """
        from trident.segmentation_models.model_zoo.tissueqc import tissue_masks, mask_to_polygons

        default, clean = tissue_masks(
            class_map,
            holes_are_tissue=holes_are_tissue,
            map_mpp=10 / self.target_mag,
            mask_mpp=self.mask_mpp,
            min_artifact_width_um=self.min_artifact_width_um,
            min_artifact_area_mm2=self.min_artifact_area_mm2,
        )
        return mask_to_polygons(
            clean if self.remove_artifacts else default,
            mask_mpp=self.mask_mpp,
            level0_mpp=level0_mpp,
            min_tissue_area_mm2=self.min_tissue_area_mm2,
            min_hole_area_mm2=self.min_artifact_area_mm2,
        )


class OtsuSegmenter(BaseSegmentationModel):
    """
    Classical image-processing tissue segmenter based on two-pass Otsu thresholding.
    """

    def __init__(self, **build_kwargs):
        super().__init__(**build_kwargs)

    def _build(self) -> Tuple[nn.Module, transforms.Compose]:
        # Keep API aligned with other segmenters used by WSI._segment_semantic.
        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 1.25
        eval_transforms = transforms.Compose([transforms.ToTensor()])
        return None, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(
                f"Input must be 4D tensor (B, C, H, W), got shape={tuple(image.shape)}"
            )

        # Lazy import: only needed for the Otsu fallback segmenter.
        import numpy as np
        from trident.segmentation_models.model_zoo.otsu import apply_otsu_thresholding

        preds = []
        for i in range(image.shape[0]):
            img_np = image[i].detach().permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

            # apply_otsu_thresholding returns background mask (1=background).
            # TRIDENT expects tissue mask (1=tissue), so invert here.
            background_mask = apply_otsu_thresholding(img_np)
            tissue_mask = (1 - background_mask).astype(np.uint8)
            preds.append(tissue_mask)

        return torch.from_numpy(np.stack(preds, axis=0))


def segmentation_model_factory(
    model_name: str, 
    confidence_thresh: float = 0.5, 
    freeze: bool = True,
    **build_kwargs,
) -> BaseSegmentationModel:
    """
    Factory function to build a segmentation model by name.
    """

    if "device" in build_kwargs:
        import warnings
        warnings.warn(
            "Passing `device` to `segmentation_model_factory` is deprecated as of version 0.1.0 "
            "Please pass `device` when segmenting the tissue, e.g., `slide.segment_tissue(..., device='cuda:0')`.",
            DeprecationWarning,
            stacklevel=2
        )

    if model_name == 'hest':
        model = HESTSegmenter(freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs)
    elif model_name == 'grandqc':
        model = GrandQCSegmenter(freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs)
    elif model_name == 'grandqc_artifact':
        model = GrandQCArtifactSegmenter(freeze=freeze, **build_kwargs)
    elif model_name == 'tissueqc':
        model = TissueQCSegmenter(freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs)
    elif model_name == 'otsu':
        model = OtsuSegmenter(freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs)
    else:
        raise ValueError(f"Model type {model_name} not supported")
    model.model_name = model_name  # canonical id for the uniform `.name` accessor
    return model
