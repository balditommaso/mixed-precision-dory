import json
import torch
import functools
from brevitas import nn as qnn
from copy import deepcopy
from typing import *
from torch import nn, tensor
from quant_strategy import *


MODULES = (
    nn.Conv2d,
    nn.Linear,
    nn.ReLU,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d
)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    
    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, quant_input: nn.Module):
        super().__init__()
        self.quant_input = quant_input
        self.model = model
        
    def forward(self, x: tensor) -> tensor:
        x = self.quant_input(x)
        return self.model(x)


def extract_modules(model: nn.Module) -> Dict[str, Any]:
    modules_config = {}
    output_channels = None
    # NOTE: Input must be quantized
    modules_config["input_quant"] = {
        "type": "Identity",
        "bit_width": 32,
        "strategy": "default",
        "implementation": "default",
        "per_channel": False,
        "output_channels": 3
    }
    
    for name, module in model.named_modules():
        # flag not handled modules
        if not isinstance(module, MODULES):
            print(f"WARNING: {name} ({module.__class__.__name__}) not extracted!")
            continue
        
        
        modules_config[name] = {
            "type": module.__class__.__name__,
            "bit_width": 32,
            "strategy": "default",
            "implementation": "default",
        }
        if isinstance(module, nn.Conv2d):
            output_channels = module.out_channels
            modules_config[name]["per_channel"] = False
        # add the possibility to have the per channel activations
        if isinstance(module, nn.ReLU):
            modules_config[name]["per_channel"] = False
            modules_config[name]["out_channels"] = output_channels
    
    return modules_config


def fold_bn_layers(model: nn.Module) -> nn.Module:
    '''
    Fold the 2D batch norm layer in the previous 2D convolution
    '''
    def _bn_folding(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
        if conv_b is None:
            conv_b = bn_rm.new_zeros(bn_rm.shape)
            
        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
        w_fold = conv_w * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
        b_fold = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
        return nn.Parameter(w_fold), nn.Parameter(b_fold)
    
    def _fold_conv_bn_eval(conv, bn):
        # assert(not (conv.training or bn.training)), "Fusion only for eval!"
        fused_conv = deepcopy(conv)
        fused_conv.weight, fused_conv.bias = _bn_folding(fused_conv.weight, fused_conv.bias,
                                bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
        return fused_conv
    
    
    new_model = deepcopy(model)
    new_model.eval()
    module_names = list(new_model._modules)
    for k, name in enumerate(module_names):
        if len(list(new_model._modules[name]._modules)) > 0:
            # iteratively re-apply the modifications
            new_model._modules[name] = fold_bn_layers(new_model._modules[name])
        else:
            if isinstance(new_model._modules[name], nn.BatchNorm2d):
                if isinstance(new_model._modules[module_names[k-1]], nn.Conv2d):
                    # folded BN
                    folded_conv = _fold_conv_bn_eval(new_model._modules[module_names[k-1]], new_model._modules[name])
                    new_model._modules[module_names[k]] = nn.Identity()
                    new_model._modules[module_names[k-1]] = folded_conv 
                    
    new_model.train()
    
    return new_model
        
        
def quantize_model(model: nn.Module, quant_config: Dict[str, Any]) -> nn.Module:
    # apply batch norm folding
    model = fold_bn_layers(model)
    
    q_model = deepcopy(model)
    
    # add input quantization layer
    q_input = get_quant_module(None, **quant_config["input_quant"])
    
    # apply quantization to the layers based on the config
    for name, config in quant_config.items():
        if name == "input_quant":
            continue
        
        module = rgetattr(q_model, name)
        q_module = get_quant_module(module, **config)
        rsetattr(q_model, name, q_module)
    
    print("[*]\t Quantization completed!")
    return QuantModel(q_model, q_input)    