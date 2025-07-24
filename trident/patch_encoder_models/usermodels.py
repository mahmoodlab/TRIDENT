from trident.patch_encoder_models.base import encoder_registry as base_encoder_registry
import torch
import torchvision.transforms as transforms
import pdb


def build_class(base_model_class):
    class DynamicModel(base_model_class):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.enc_name = self.enc_name + '_green'

        def _build(self):

            model, eval_transforms, precision = super()._build()

            start_i=[i for i,t in enumerate(eval_transforms.transforms) if isinstance(t, transforms.ToTensor)]
            if len(start_i) == 0:
                raise('Cannot find ToTensor in patch_transforms')
            start_i = start_i[0]
            eval_transforms = transforms.Compose(eval_transforms.transforms[0:start_i+1] #ToTensor()                                                                                                     
                                                    + [BoostGreen()]
                                                    + eval_transforms.transforms[start_i+1:])
            return model, eval_transforms, precision
    return DynamicModel

encoder_registry = {}                    
for name, model in base_encoder_registry.items():
        model=build_class(model)
        encoder_registry[name+"_green"] = model

class BoostGreen:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """                                                                                                                                                                                                
        Assumes input is a tensor image with shape (C, H, W) and values in [0, 1].                                                                                                                         
        Boosts the green channel by 20%, clips at 1.0.                                                                                                                                                     
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")

        img = img.clone()  # avoid in-place modification                                                                                                                                                   
        img[1] = torch.clamp(img[1] * 1.2, max=1.0)
        return img


