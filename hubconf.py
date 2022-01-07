from torch import nn
import torch
from timm.models.vision_transformer import deit_tiny_patch16_224
dependencies = ['torch', "timm"]


def birds_325_deit_tiny_patch16_224(pretrained=True):
    """Loading function for the pretrained birds_325 model.
    The model has a accuracy of 97% (1555/1600) on the test set.

    :param pretrained: Whether to load the pretrained bird model, defaults to True
    :type pretrained: bool, optional
    :return: a pretrained birds_325 model
    :rtype: torch.nn.Module
    """
    loaded = torch.jit.load("birds-325-deit-tiny-patch16-224.zip")
    model = deit_tiny_patch16_224(pretrained=False)
    n_inputs = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(n_inputs, 2048),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, 325) # 325 classes
    )
    if pretrained:
        state_dict = loaded.state_dict()
        model.load_state_dict(state_dict)
    return model
