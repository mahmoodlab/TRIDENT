import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import traceback
from abc import abstractmethod
from huggingface_hub import snapshot_download

from trident.IO import get_dir, get_weights_path

class SegmentationModel(torch.nn.Module):
    def __init__(self, freeze=True, confidence_thresh=0.5, **build_kwargs):
        super().__init__()
        self.model, self.eval_transforms = self._build(**build_kwargs)
        self.confidence_thresh = confidence_thresh

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            
    def forward(self, batch):
        '''
        Can be overwritten if model requires special forward pass.
        '''
        z = self.model(batch)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass


class HESTSegmenter(SegmentationModel):
    def _build(self):
        self.input_size = 512
        self.precision = torch.float16
        self.target_mag = 10

        MODEL_TD_NAME = 'deeplabv3_seg_v4.ckpt'
        weights_path = get_weights_path('seg', 'hest')
        
        try:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
        except:
            traceback.print_exc()
            raise Exception(
                "Failed to download PyTorch Vision or load deeplabv3_resnet50 from cache.\n"
                "Make sure you have internet access or you have pre-downloaded the files into the PyTorch cache directory.\n"
                "Run `git clone --branch v0.10.0 --depth 1 https://github.com/pytorch/vision.git pytorch_vision_v0.10.0` and"
                "place in the folder specified by the PyTorch cache directory, `torch.hub.get_dir()`"
            )

        model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=2,
            kernel_size=1,
            stride=1
        )

        if weights_path != "":
            pass
        
        else:
            try:
                checkpoint_dir = snapshot_download(
                    repo_id="MahmoodLab/hest-tissue-seg",
                    repo_type='model'
                )
                weights_path = os.path.join(checkpoint_dir, MODEL_TD_NAME)
                
            except:
                traceback.print_exc()
                raise Exception("Failed to download HEST model, make sure that you were granted access and that you correctly registered your token")

        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)

        clean_state_dict = {}
        for key in checkpoint['state_dict']:
            if 'aux' in key:
                continue
            new_key = key.replace('model.', '')
            clean_state_dict[new_key] = checkpoint['state_dict'][key]
        model.load_state_dict(clean_state_dict)

        eval_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        return model, eval_transforms
    
    def forward(self, image):
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

    def _build(self):
        """
        Credit to https://www.nature.com/articles/s41467-024-54769-y
        """
        import segmentation_models_pytorch as smp

        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 10

        MODEL_TD_NAME = 'GrandQC_MPP1_state_dict.pth'
        ENCODER_MODEL_TD = 'timm-efficientnet-b0'
        ENCODER_MODEL_TD_WEIGHTS = 'imagenet'

        weights_path = get_weights_path('seg', 'grandqc_remove_artifacts')

        model = smp.Unet(
            encoder_name=ENCODER_MODEL_TD,
            encoder_weights=ENCODER_MODEL_TD_WEIGHTS,
            classes=8,
            activation=None,
        )

        if weights_path != "":
            pass
        
        else:
            try:
                checkpoint_dir = snapshot_download(
                    repo_id="MahmoodLab/hest-tissue-seg",
                    repo_type='model'
                )
                weights_path = os.path.join(checkpoint_dir, MODEL_TD_NAME)
                
            except:
                traceback.print_exc()
                raise Exception("Failed to download HEST model, make sure that you were granted access and that you correctly registered your token")

        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

        # eval_transforms = smp.encoders.get_preprocessing_fn(ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS)  # to double check if same
        eval_transforms = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return model, eval_transforms

    def forward(self, batch):
        '''
        Custom forward pass.
        '''
        logits = self.model.predict(batch)
        probs = torch.softmax(logits, dim=1)  
        _, predicted_classes = torch.max(probs, dim=1)  
        predictions = torch.where(predicted_classes > 1, 0, 1)
        predictions = predictions.to(torch.uint8)
        return predictions


class GrandQCSegmenter(SegmentationModel):

    def _build(self):
        """
        Credit to https://www.nature.com/articles/s41467-024-54769-y
        """
        import segmentation_models_pytorch as smp

        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 1

        MODEL_TD_NAME = 'Tissue_Detection_MPP10.pth'
        ENCODER_MODEL_TD = 'timm-efficientnet-b0'
        ENCODER_MODEL_TD_WEIGHTS = 'imagenet'

        weights_path = get_weights_path('seg', 'grandqc')

        model = smp.UnetPlusPlus(
            encoder_name=ENCODER_MODEL_TD,
            encoder_weights=ENCODER_MODEL_TD_WEIGHTS,
            classes=2,
            activation=None,
        )

        if weights_path != "":
            pass
        
        else:
            try:
                checkpoint_dir = snapshot_download(
                    repo_id="MahmoodLab/hest-tissue-seg",
                    repo_type='model'
                )
                weights_path = os.path.join(checkpoint_dir, MODEL_TD_NAME)
                
            except:
                traceback.print_exc()
                raise Exception("Failed to download HEST model, make sure that you were granted access and that you correctly registered your token")

        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))

        # eval_transforms = smp.encoders.get_preprocessing_fn(ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS)
        eval_transforms = transforms.Compose([
            JpegCompressionTransform(quality=80),
            transforms.ToTensor(),  # Converts to [0,1] range and moves channels to [C, H, W]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return model, eval_transforms

    def forward(self, batch):
        '''
        Custom forward pass.
        '''
        logits = self.model.predict(batch)
        probs = torch.softmax(logits, dim=1)  
        max_probs, predicted_classes = torch.max(probs, dim=1)  
        predictions = (max_probs >= self.confidence_thresh) * (1 - predicted_classes)
        predictions = predictions.to(torch.uint8)
 
        return predictions


def segmentation_model_factory(model_name, freeze=True, confidence_thresh=0.5):
    '''
    Build a slide encoder based on model name.
    '''

    checkpoint_dir = get_dir()
    
    if model_name == 'hest':
        return HESTSegmenter(freeze, confidence_thresh=confidence_thresh)
    elif model_name == 'grandqc':
        return GrandQCSegmenter(freeze, confidence_thresh=confidence_thresh)
    elif model_name == 'grandqc_artifact':
        return GrandQCArtifactSegmenter(freeze)
    else:
        raise ValueError(f"Model type {model_name} not supported")
