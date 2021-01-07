from .losses import *

__all__ = [
    'BinaryDiceLoss', 'DiceLoss', 'FocalLoss'
]


def build_criterion():
    # to do add paramaters
    criterion = DiceLoss()
    return criterion