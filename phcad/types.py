"""Function signatures"""

from typing import Protocol

from torch import Tensor
from torchvision.transforms.v2 import Transform


class Preprocessor(Protocol):
    """Function for preprocessing data before use in a model"""
    def __call__(self, input: Tensor, **kwargs) -> Tensor: ...

class AugmentableTransform(Protocol):
    """If base is true return preprocessing transform, else return preprocessed and
    augmented input"""
    def __call__(self, base: bool, **kwargs) -> Transform | Preprocessor: ...
