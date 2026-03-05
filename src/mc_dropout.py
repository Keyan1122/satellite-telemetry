import torch.nn as nn


def enable_mc_dropout(model):
    """
    Enable Monte Carlo Dropout during inference.

    This function sets all Dropout layers to train mode
    while keeping the rest of the model in eval mode.
    """

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
