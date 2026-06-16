# in submodule
from trident.slide_encoder_models.load import (
    encoder_registry,
    encoder_factory,
    MeanSlideEncoder,
    ABMILSlideEncoder,
    PRISMSlideEncoder,
    CHIEFSlideEncoder,
    GigaPathSlideEncoder,
    TitanSlideEncoder,
    ThreadsSlideEncoder,
    MadeleineSlideEncoder,
    FeatherSlideEncoder,
    FeatherUni2SlideEncoder,
    CARESlideEncoder
)

__all__ = [
    "encoder_registry",
    "encoder_factory",
    "TitanSlideEncoder",
    "ThreadsSlideEncoder",
    "MadeleineSlideEncoder",
    "MeanSlideEncoder",
    "ABMILSlideEncoder",
    "PRISMSlideEncoder",
    "CHIEFSlideEncoder",
    "GigaPathSlideEncoder",
    "FeatherSlideEncoder",
    "FeatherUni2SlideEncoder",
    "CARESlideEncoder"
]