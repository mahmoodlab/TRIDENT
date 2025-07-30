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
    FeatherSlideEncoder
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
    "FeatherSlideEncoder"
]