import csv
import os
from dataclasses import dataclass
from math import ceil, log2, prod
from typing import Optional, Sequence, Tuple

import numpy as np
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name

from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation


MAX_QUANT_HOPS = 3
DEFAULT_BITWIDTH = 8
DEFAULT_BIAS_BITWIDTH = 32
DEFAULT_ACC_BITWIDTH = 32
QUANT_OPS = {"Quant", "Trunc"}


@dataclass(frozen=True)
class ProfileMetrics:
    """Coarse, platform-independent metrics for one operator."""
    macs: int
    memory_bytes: int
    bops: int


def _static_shape(model: ModelWrapper, tensor_name: str) -> Tuple[int, ...]:
    shape = model.get_tensor_shape(tensor_name)
    if shape is None:
        raise ValueError(f"Shape is unavailable for tensor '{tensor_name}'.")
    if any(dim is None for dim in shape):
        raise ValueError(f"Tensor '{tensor_name}' has a dynamic shape: {shape}.")
    return tuple(int(dim) for dim in shape)


def _num_elements(shape: Sequence[int]) -> int:
    return int(prod(int(dim) for dim in shape))


def _bits_to_bytes(num_values: int, bitwidth: int) -> int:
    """Return packed storage in bytes, rounded up for sub-byte tensors."""
    return int(ceil((int(num_values) * int(bitwidth)) / 8))


def _tensor_bytes(shape: Sequence[int], bitwidth: int) -> int:
    return _bits_to_bytes(_num_elements(shape), bitwidth)


def _initializer_scalar(model: ModelWrapper, initializer_name: str) -> int:
    initializer = model.get_initializer(initializer_name)
    if initializer is None:
        raise ValueError(f"Initializer '{initializer_name}' was not found.")

    values = np.asarray(initializer).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Initializer '{initializer_name}' is empty.")
    return int(values[0])


def _initializer_size(model: ModelWrapper, initializer_name: str) -> int:
    initializer = model.get_initializer(initializer_name)
    if initializer is None:
        return 0
    return int(np.asarray(initializer).size)


def _string_attribute(node: NodeProto, name: str, default: str) -> str:
    attribute = get_by_name(node.attribute, name)
    if attribute is None:
        return default
    return attribute.s.decode("utf-8")


def _int_attribute(node: NodeProto, name: str, default: int) -> int:
    attribute = get_by_name(node.attribute, name)
    if attribute is None:
        return default
    return int(attribute.i)


def _ints_attribute(
    node: NodeProto,
    name: str,
    default: Optional[Sequence[int]] = None,
) -> Optional[Tuple[int, ...]]:
    attribute = get_by_name(node.attribute, name)
    if attribute is None:
        return None if default is None else tuple(int(value) for value in default)
    return tuple(int(value) for value in attribute.ints)


def get_implementation(node: NodeProto) -> str:
    return _string_attribute(node, "implementation", "default")


def find_upstream_quant(
    model: ModelWrapper,
    tensor_name: str,
    max_hops: int = MAX_QUANT_HOPS,
) -> Optional[str]:
    """Return the bitwidth initializer of the closest upstream quantizer."""
    current_tensor = tensor_name
    visited = set()

    for _ in range(max_hops):
        if current_tensor in visited:
            return None
        visited.add(current_tensor)

        node = model.find_producer(current_tensor)
        if node is None:
            return None

        if node.op_type in QUANT_OPS and len(node.input) > 3:
            return node.input[3]

        if not node.input:
            return None
        current_tensor = node.input[0]

    return None


def find_downstream_quant(
    model: ModelWrapper,
    tensor_name: str,
    max_hops: int = MAX_QUANT_HOPS,
) -> Optional[str]:
    """Return the bitwidth initializer of the closest downstream quantizer."""
    current_tensor = tensor_name
    visited = set()

    for _ in range(max_hops):
        if current_tensor in visited:
            return None
        visited.add(current_tensor)

        node = model.find_consumer(current_tensor)
        if node is None:
            return None

        if node.op_type in QUANT_OPS and len(node.input) > 3:
            return node.input[3]

        if not node.output:
            return None
        current_tensor = node.output[0]

    return None


def _upstream_bitwidth(model: ModelWrapper, tensor_name: str, default: int = DEFAULT_BITWIDTH) -> int:
    initializer_name = find_upstream_quant(model, tensor_name)
    if initializer_name is None:
        return int(default)
    return _initializer_scalar(model, initializer_name)


def _downstream_bitwidth(model: ModelWrapper, tensor_name: str, default: int = DEFAULT_BITWIDTH) -> int:
    initializer_name = find_downstream_quant(model, tensor_name)
    if initializer_name is None:
        return int(default)
    return _initializer_scalar(model, initializer_name)


def _bias_bytes(model: ModelWrapper, node: NodeProto) -> int:
    if len(node.input) < 3 or not node.input[2]:
        return 0
    bias_shape = _static_shape(model, node.input[2])
    return _tensor_bytes(bias_shape, DEFAULT_BIAS_BITWIDTH)


def _lut_bytes(input_bitwidth: int, weight_bitwidth: int, output_bitwidth: int) -> int:
    entries = 1 << (int(input_bitwidth) + int(weight_bitwidth))
    return _bits_to_bytes(entries, output_bitwidth)


def _mac_bops(
    macs: int,
    input_bitwidth: int,
    weight_bitwidth: int,
    output_bitwidth: int,
) -> int:
    """Preserve the original gross BOP proxy for MAC-based operators."""
    return int(
        macs
        * (
            1
            + int(input_bitwidth)
            + int(weight_bitwidth)
            + int(output_bitwidth)
        )
    )


def profile_convolution(model: ModelWrapper, node: NodeProto) -> ProfileMetrics:
    input_shape = _static_shape(model, node.input[0])
    weight_shape = _static_shape(model, node.input[1])
    output_shape = _static_shape(model, node.output[0])

    if len(weight_shape) != 4:
        raise ValueError(f"Expected a 4-D convolution weight, got {weight_shape}.")

    input_bitwidth = _upstream_bitwidth(model, node.input[0])
    weight_bitwidth = _upstream_bitwidth(model, node.input[1])
    output_bitwidth = DEFAULT_ACC_BITWIDTH
    padding = _ints_attribute(node, "pads")
    stride = _ints_attribute(node, "strides")
    _, C_in, H_in, W_in = input_shape
    C_out, _, K_h, K_w = weight_shape
    p_t, p_b, p_l, p_r = padding
    s1, s2 = stride
    H_out = (H_in + p_t + p_b - K_h) // s1 + 1
    W_out = (W_in + p_l + p_r - K_w) // s2 + 1

    total_mem = (
        _tensor_bytes(input_shape, input_bitwidth) 
        + _tensor_bytes(weight_shape, weight_bitwidth) 
        + _tensor_bytes(output_shape, output_bitwidth) 
        + _bias_bytes(model, node)
    )
    macs = C_out * H_out * W_out * C_in * K_h * K_w
    
    if get_implementation(node) == "lut":
        # LUT-based implementation adds additional lookup table storage
        total_mem += _lut_bytes(input_bitwidth, weight_bitwidth, output_bitwidth)
        macs = 0
    else:
        # im2col matrix dimensions: [C_in * K_h * K_w, H_out * W_out]
        im2col_elems = C_in * K_h * K_w * H_out * W_out
        im2col_mem = _bits_to_bytes(im2col_elems, input_bitwidth)
        total_mem += im2col_mem

    return ProfileMetrics(
        macs=int(macs),
        memory_bytes=int(total_mem),
        bops=_mac_bops(
            C_out * H_out * W_out * C_in * K_h * K_w,
            input_bitwidth,
            weight_bitwidth,
            output_bitwidth,
        ),
    )


def profile_gemm(model: ModelWrapper, node: NodeProto) -> ProfileMetrics:
    input_shape = _static_shape(model, node.input[0])
    weight_shape = _static_shape(model, node.input[1])
    output_shape = _static_shape(model, node.output[0])

    input_bitwidth = _upstream_bitwidth(model, node.input[0])
    weight_bitwidth = _upstream_bitwidth(model, node.input[1])
    output_bitwidth = DEFAULT_ACC_BITWIDTH

    if node.op_type == "Gemm" and _int_attribute(node, "transA", 0):
        reduction_size = input_shape[-2]
    else:
        reduction_size = input_shape[-1]

    macs = _num_elements(output_shape) * reduction_size
    total_mem = (
        _tensor_bytes(input_shape, input_bitwidth)
        + _tensor_bytes(weight_shape, weight_bitwidth)
        + _tensor_bytes(output_shape, output_bitwidth)
        + _bias_bytes(model, node)
    )

    if get_implementation(node) == "lut":
        total_mem += _lut_bytes(
            input_bitwidth,
            weight_bitwidth,
            output_bitwidth,
        )

    return ProfileMetrics(
        macs=int(macs),
        memory_bytes=int(total_mem),
        bops=_mac_bops(
            macs,
            input_bitwidth,
            weight_bitwidth,
            output_bitwidth,
        ),
    )


def profile_relu(model: ModelWrapper, node: NodeProto) -> ProfileMetrics:
    input_shape = _static_shape(model, node.input[0])
    output_shape = _static_shape(model, node.output[0])

    input_bitwidth = _upstream_bitwidth(model, node.input[0])
    output_bitwidth = _downstream_bitwidth(
        model,
        node.output[0],
        default=input_bitwidth,
    )

    output_elements = _num_elements(output_shape)
    memory_bytes = (
        _tensor_bytes(input_shape, input_bitwidth)
        + _tensor_bytes(output_shape, output_bitwidth)
    )
    bops = output_elements * (input_bitwidth + 1)

    return ProfileMetrics(0, int(memory_bytes), int(bops))


def profile_quant(model: ModelWrapper, node: NodeProto) -> ProfileMetrics:
    if model.find_producer(node.input[0]) is None:
        raise ValueError()
    
    input_shape = _static_shape(model, node.input[0])
    output_shape = _static_shape(model, node.output[0])

    input_bitwidth = _upstream_bitwidth(model, node.input[0])
    if len(node.input) > 3:
        output_bitwidth = _initializer_scalar(model, node.input[3])
    else:
        output_bitwidth = _downstream_bitwidth(
            model,
            node.output[0],
            default=input_bitwidth,
        )

    scale_count = _initializer_size(model, node.input[1]) if len(node.input) > 1 else 1
    scale_count = max(scale_count, 1)
    output_elements = _num_elements(output_shape)

    memory_bytes = (
        _tensor_bytes(input_shape, input_bitwidth)
        + _tensor_bytes(output_shape, output_bitwidth)
    )

    if get_implementation(node) == "thresholds":
        num_thresholds = max((1 << output_bitwidth) - 1, 0)
        search_steps = max(1, int(ceil(log2(max(num_thresholds, 2)))))
        bops = output_elements * search_steps * input_bitwidth
        parameter_bytes = _bits_to_bytes(
            num_thresholds * scale_count,
            input_bitwidth,
        )
    else:
        bops = output_elements * (2 * input_bitwidth + 5)
        parameter_bytes = _bits_to_bytes(scale_count, 32)

    return ProfileMetrics(
        macs=0,
        memory_bytes=int(memory_bytes + parameter_bytes),
        bops=int(bops),
    )


def _pool_kernel_shape(model: ModelWrapper, node: NodeProto) -> Tuple[int, ...]:
    input_shape = _static_shape(model, node.input[0])

    if node.op_type.startswith("Global"):
        if len(input_shape) < 3:
            raise ValueError(
                f"Global pooling expects spatial dimensions, got {input_shape}."
            )
        return tuple(input_shape[2:])

    kernel_shape = _ints_attribute(node, "kernel_shape")
    if kernel_shape is None:
        raise ValueError(f"Missing kernel_shape for node '{node.name or node.op_type}'.")
    return kernel_shape



def _pool_memory(
    model: ModelWrapper,
    node: NodeProto,
) -> Tuple[int, int, int]:
    input_shape = _static_shape(model, node.input[0])
    output_shape = _static_shape(model, node.output[0])

    input_bitwidth = _upstream_bitwidth(model, node.input[0])
    output_bitwidth = _downstream_bitwidth(
        model,
        node.output[0],
        default=input_bitwidth,
    )

    memory_bytes = (
        _tensor_bytes(input_shape, input_bitwidth)
        + _tensor_bytes(output_shape, output_bitwidth)
    )
    return int(memory_bytes), int(input_bitwidth), _num_elements(output_shape)


def profile_avgpool(model: ModelWrapper, node: NodeProto) -> ProfileMetrics:
    kernel_elements = _num_elements(_pool_kernel_shape(model, node))
    memory_bytes, input_bitwidth, output_elements = _pool_memory(model, node)

    bops = output_elements * kernel_elements * input_bitwidth
    return ProfileMetrics(0, memory_bytes, int(bops))


def profile_maxpool(model: ModelWrapper, node: NodeProto) -> ProfileMetrics:
    kernel_elements = _num_elements(_pool_kernel_shape(model, node))
    memory_bytes, input_bitwidth, output_elements = _pool_memory(model, node)

    comparisons_per_output = max(kernel_elements - 1, 0)
    bops = output_elements * comparisons_per_output * input_bitwidth
    return ProfileMetrics(0, memory_bytes, int(bops))


def profile_add(model: ModelWrapper, node: NodeProto) -> ProfileMetrics:
    input_a_shape = _static_shape(model, node.input[0])
    input_b_shape = _static_shape(model, node.input[1])
    output_shape = _static_shape(model, node.output[0])

    input_a_bitwidth = _upstream_bitwidth(model, node.input[0])
    input_b_bitwidth = _upstream_bitwidth(model, node.input[1])
    output_bitwidth = _downstream_bitwidth(
        model,
        node.output[0],
        default=max(input_a_bitwidth, input_b_bitwidth),
    )

    memory_bytes = (
        _tensor_bytes(input_a_shape, input_a_bitwidth)
        + _tensor_bytes(input_b_shape, input_b_bitwidth)
        + _tensor_bytes(output_shape, output_bitwidth)
    )

    bits_per_add = max(input_a_bitwidth, input_b_bitwidth) + 1
    bops = _num_elements(output_shape) * bits_per_add

    return ProfileMetrics(0, int(memory_bytes), int(bops))


def profile_node(model: ModelWrapper,node: NodeProto) -> Optional[ProfileMetrics]:
    if node.op_type == "Conv":
        return profile_convolution(model, node)
    if node.op_type in {"Gemm", "MatMul", "Matmul"}:
        return profile_gemm(model, node)
    if node.op_type in QUANT_OPS:
        try:
            return profile_quant(model, node)
        except:
            return None
    if node.op_type == "Relu":
        return profile_relu(model, node)
    if node.op_type in {"GlobalAveragePool", "AveragePool"}:
        return profile_avgpool(model, node)
    if node.op_type in {"GlobalMaxPool", "MaxPool"}:
        return profile_maxpool(model, node)
    if node.op_type == "Add":
        return profile_add(model, node)
    return None


def _layer_name(node: NodeProto, index: int) -> str:
    aliases = {
        "GlobalAveragePool": "avgpool",
        "AveragePool": "avgpool",
        "GlobalMaxPool": "maxpool",
        "MaxPool": "maxpool",
    }
    op_name = aliases.get(node.op_type, node.op_type)
    return f"{op_name}_{index}"



class ImplementationAwareTrace(BaseTrasformation):
    """
    Write a coarse operator trace without platform-specific performance modeling.

    The trace reports mathematical MACs, a gross BOP proxy, logical tensor and
    parameter storage in bytes, and the implementation label stored in the graph.
    """

    def __init__(self, output_path: str, file_name: str, verbose: bool = False):
        self.output_path = output_path
        self.file_name = file_name
        super().__init__(verbose)


    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        csv_path = os.path.join(self.output_path, f"{self.file_name}.csv")
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        profiled_index = 0
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["Op.", "MACs", "Memory", "BOPs", "implementation"]
            )

            for node in model.graph.node:
                try:
                    metrics = profile_node(model, node)
                except (IndexError, TypeError, ValueError) as error:
                    if self.verbose:
                        node_name = node.name or node.op_type
                        print(f"Skipping '{node_name}': {error}")
                    continue

                if metrics is None:
                    continue

                writer.writerow(
                    [
                        _layer_name(node, profiled_index),
                        metrics.macs,
                        metrics.memory_bytes,
                        metrics.bops,
                        get_implementation(node),
                    ]
                )
                profiled_index += 1

        return model, False