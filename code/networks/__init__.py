from .NestedUNet import *
from .unet import *

import torch

__model_factory = {
    'Unet': UNet,
    'UNetVGG': UNetVGG,
    'NestUnet': NestedUNet
}


def build_model(model_name, num_classes, input_channels):
    avail_models = list(__model_factory.keys())
    if model_name not in avail_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(model_name, avail_models)
        )

    model_kwargs = dict(num_classes=num_classes, input_channels=input_channels)
    model = __model_factory[model_name](**model_kwargs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)
    return model