import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from abc import abstractmethod

from trident.IO import get_dir, get_weights_path, has_internet_connection


class SegmentationModel(torch.nn.Module):

    _has_internet = has_internet_connection()

    def __init__(self, freeze=True, confidence_thresh=0.5, **build_kwargs):
        """
        Initialize Segmentation model wrapper.

        Args:
            freeze (bool, optional): If True, the model's parameters are frozen 
                (i.e., not trainable) and the model is set to evaluation mode. 
                Defaults to True.
            confidence_thresh (float, optional): Threshold for prediction confidence. 
                Predictions below this threshold may be filtered out or ignored. 
                Default is 0.5. Set to 0.4 to keep more tissue.
            **build_kwargs: Additional keyword arguments passed to the internal 
                `_build` method.

        Attributes:
            model (torch.nn.Module): The constructed model.
            eval_transforms (Callable): Transformations to apply to input data during inference.
        """
        super().__init__()
        self.model, self.eval_transforms = self._build(**build_kwargs)
        self.confidence_thresh = confidence_thresh

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            
    def forward(self, image):
        """
        Can be overwritten if model requires special forward pass.
        """
        z = self.model(image)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs) -> tuple[nn.Module, transforms.Compose]:
        """
        Build the segmentation model and preprocessing transforms.
        """
        pass


class HESTSegmenter(SegmentationModel):

    def __init__(self, **build_kwargs):
        """
        HESTSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
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
            if not SegmentationModel._has_internet:
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


class GrandQCArtifactSegmenter(SegmentationModel):

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
        encoder_weights = 'imagenet'
        weights_path = get_weights_path('seg', 'grandqc_artifact')

        # Verify that user-provided weights_path is valid
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Expected checkpoint at '{weights_path}', but the file was not found."
            )

        # Initialize model
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=8,
            activation=None,
        )

        # Attempt to download if file is missing and not already available
        if not weights_path:
            if not SegmentationModel._has_internet:
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


class GrandQCSegmenter(SegmentationModel):
    
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
        encoder_weights = 'imagenet'
        weights_path = get_weights_path('seg', 'grandqc') 

        # Verify that user-provided weights_path is valid
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Expected checkpoint at '{weights_path}', but the file was not found."
            )

        # Verify checkpoint path
        if not weights_path:
            if not SegmentationModel._has_internet:
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
            encoder_weights=encoder_weights,
            classes=2,
            activation=None,
        )

        # Load checkpoint
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

        # Model config
        self.input_size = 512
        self.precision = torch.float32
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


def segmentation_model_factory(
    model_name: str, 
    confidence_thresh: float = 0.5, 
    freeze: bool = True,
    **build_kwargs,
) -> SegmentationModel:
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
        return HESTSegmenter(freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs)
    elif model_name == 'grandqc':
        return GrandQCSegmenter(freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs)
    elif model_name == 'grandqc_artifact':
        return GrandQCArtifactSegmenter(freeze=freeze, **build_kwargs)
    else:
        raise ValueError(f"Model type {model_name} not supported")
