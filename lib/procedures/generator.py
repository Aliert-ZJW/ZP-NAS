from layers import layers as layers
import torch

def trainable(module):
    r"""Returns boolean whether a module is trainable.
    """
    return not isinstance(module, (layers.Identity1d, layers.Identity2d))

def prunable(module, batchnorm, residual):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (layers.Linear, layers.Conv2d))
    print(isprunable)
    if batchnorm:
        isprunable |= isinstance(module, (layers.BatchNorm1d, layers.BatchNorm2d))
    if residual:
        isprunable |= isinstance(module, (layers.Identity1d, layers.Identity2d))
    return isprunable

def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for module in filter(lambda p: trainable(p), model.modules()):
        for param in module.parameters(recurse=False):
            yield param

def masked_parameters(model, bias=False, batchnorm=False, residual=False):  # all False
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    # for i in model.modules():
    #     print(i)
    # for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
    #     for param in module.parameters(recurse=False):
    #         if param is not module.bias or bias is True:
    # #             yield
    # for layer_ in model.named_modules():
    #     # print('layer', layer_)
    # signs = {}
    # for name in model.state_dict().items():  # conv and linear param
    #     # signs[name] = torch.sign(param)
    #     # print('signs', name)
    #     # param.abs_()
    # return signs
    # for module in model.modules():
    for name, param in model.named_parameters():
        if 'weight' in name:
            yield name, param



