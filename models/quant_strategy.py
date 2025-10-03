from torch import nn
from brevitas import nn as qnn
from brevitas.quant.scaled_int import (
    Int32Bias,
    Int16Bias,
    Int8WeightPerTensorFloat,
    Int8WeightPerChannelFloat,
    Int8ActPerTensorFloat,
    Uint8ActPerTensorFloat
)
# 2-bit quantization
from brevitas.quant.ternary import (
    SignedTernaryActPerTensorConst,
    SignedTernaryWeightPerTensorConst
)

# 4-bit quantization
class Int4WeightPerTensorFloat(Int8WeightPerTensorFloat):
    scaling_min_val = 2e-16
    bit_width = 4

class Int4WeightPerChannelFloat(Int8WeightPerChannelFloat):
    scaling_min_val = 2-16
    bit_width = 4

class Int4ActPerTensorFloat(Int8ActPerTensorFloat):
    scaling_min_val = 2-16
    bit_width = 4
    
class Uint4ActPerTensorFloat(Uint8ActPerTensorFloat):
    scaling_min_val = 2-16
    bit_width = 4
    
    
def get_quantizer(kind: str, bit_width: int, mode: str = "per_tensor"):
    """
    kind: "weight" or "act"
    bit_width: 2, 4, or 8
    mode:
      - for weights: "per_tensor" or "per_channel"
      - for activations: "signed" or "unsigned"
    """
    mapping = {
        # ---- BIAS ----
        ("bias", 4, "per_tensor"): Int16Bias,
        ("bias", 8, "per_tensor"): Int32Bias,
        # ---- WEIGHTS ----
        ("weight", 8, "per_tensor"): Int8WeightPerTensorFloat,
        ("weight", 8, "per_channel"): Int8WeightPerChannelFloat,
        ("weight", 4, "per_tensor"): Int4WeightPerTensorFloat,
        ("weight", 4, "per_channel"): Int4WeightPerChannelFloat,
        ("weight", 2, "per_tensor"): SignedTernaryWeightPerTensorConst,  # only per-tensor
        # ---- ACTIVATIONS ----
        ("act", 8, "signed"): Int8ActPerTensorFloat,
        ("act", 8, "unsigned"): Uint8ActPerTensorFloat,
        ("act", 4, "signed"): Int4ActPerTensorFloat,
        ("act", 4, "unsigned"): Uint4ActPerTensorFloat,
        ("act", 2, "signed"): SignedTernaryActPerTensorConst,  # ternary is signed
        ("act", 2, "unsigned"): None  # not supported
    }

    quantizer = mapping.get((kind, bit_width, mode))
    if quantizer is None:
        raise ValueError(f"No quantizer available for {kind}, {bit_width}-bit, mode={mode}")
    return quantizer

    

def get_quant_module(module: nn.Module, type: str, bit_width: int, strategy: str, **args) -> nn.Module:
    q_module = None
    if isinstance(module, nn.Conv2d):
        q_module = qnn.QuantConv2d(
            module.in_channels, 
            module.out_channels, 
            module.kernel_size,
            groups=module.groups,
            stride=module.stride,
            padding=module.padding,
            bias=module.bias is not None and bit_width > 2,
            bias_quant=get_quantizer("bias", bit_width),
            weight_quant=get_quantizer("weight", bit_width, "per_channel" if args["per_channel"] else "per_tensor"),
            scaling_min_val = 1e-6
        )
    elif isinstance(module, nn.Linear):
        out_features, in_features = module.weight.shape
        q_module = qnn.QuantLinear(
            in_features,
            out_features,
            bias=module.bias is not None and bit_width > 2,
            bias_quant=get_quantizer("bias", bit_width),
            weight_quant=get_quantizer("weight", bit_width),
            scaling_min_val = 1e-6
        )
    
    elif isinstance(module, nn.ReLU):
        config = {}
        # per channel activations
        if args["per_channel"]:
            config = {
                "per_channel_broadcastable_shape": (1, args["out_channels"], 1, 1),
                "scaling_stats_permute_dims": (1, 0, 2, 3),
                "scaling_per_output_channel": True,
            }
        q_module = qnn.QuantReLU(
            act_quant=get_quantizer("act", bit_width, "unsigned"),
            return_quant_tensor=True,
            scaling_min_val=1e-6,
            **config
        )
    
    elif isinstance(module, nn.AvgPool2d):
        q_module = qnn.TruncAvgPool2d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            bit_width=bit_width
        )
    
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        q_module = qnn.TruncAdaptiveAvgPool2d(
            output_size=module.output_size,
            bit_width=bit_width
        )
    
    elif type == "Identity":
        q_module = qnn.QuantIdentity(
            act_quant=get_quantizer("act", bit_width, "signed"),
            return_quant_tensor=True
        )
    
    # load the parameters in the quantized version of the module
    if module is not None:
        state_dict = module.state_dict()
        q_module.load_state_dict(state_dict)
        
    return q_module