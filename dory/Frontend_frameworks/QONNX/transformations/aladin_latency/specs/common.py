from __future__ import annotations

from typing import Any, Dict, List

from ..config import AutoSpecConfig
from ..descriptors import DMATransferSpec, NodeExecutionSpec
from ..utils import align_up, nonnegative_int


def is_supported_pulp_conv_node(node: Any) -> bool:
    op_type = str(getattr(node, 'op_type', '')).lower()
    required = ('input_dimensions', 'output_dimensions', 'input_channels', 'output_channels', 'kernel_shape', 'strides', 'MACs')
    return 'conv' in op_type and all((hasattr(node, name) for name in required))


def is_supported_pool_node(node: Any) -> bool:
    op_type = str(getattr(node, 'op_type', '')).lower()
    required = ('input_dimensions', 'output_dimensions', 'kernel_shape', 'strides')
    return 'pool' in op_type and all(hasattr(node, name) for name in required)


def is_supported_linear_node(node: Any) -> bool:
    op_type = str(getattr(node, 'op_type', '')).lower()
    name = str(getattr(node, 'name', '')).lower()
    return any(token in op_type or token in name for token in (
        'fullyconnected', 'fully_connected', 'linear', 'gemm'
    ))


def is_supported_add_node(
    node: Any,
    kernel_name: Union[str, None] = None,
) -> bool:
    op_type = str(getattr(node, 'op_type', '')).lower()
    name = str(getattr(node, 'name', '')).lower()
    kernel = str(kernel_name or '').lower()
    return (
        'pulp_nn_add_' in kernel
        or op_type in ('add', 'sum', 'addition', 'reluadd', 'reluaddition')
        or 'addition' in op_type
        or name.startswith('add')
        or 'reluadd' in name
        or 'addition' in name
    )


def infer_generic_l1_offsets(node: Any, config: AutoSpecConfig) -> Dict[str, int]:
    alignment = config.l1_alignment_bytes
    guard = config.l1_guard_bytes
    input_size = nonnegative_int(getattr(node, 'input_activation_memory', 0))
    output_size = nonnegative_int(getattr(node, 'output_activation_memory', 0))
    weight_size = nonnegative_int(getattr(node, 'weight_memory', 0))
    bias_size = nonnegative_int(getattr(node, 'bias_memory', 0))
    offsets: Dict[str, int] = {'input': 0}
    cursor = input_size
    offsets['output'] = align_up(cursor + guard, alignment)
    cursor = offsets['output'] + output_size
    if weight_size:
        offsets['weights'] = align_up(cursor + guard, alignment)
        cursor = offsets['weights'] + weight_size
    if bias_size:
        offsets['bias'] = align_up(cursor + guard, alignment)
    return offsets


def default_execution_spec(node: Any) -> NodeExecutionSpec:
    transfers: List[DMATransferSpec] = []
    for name, size in (('input', getattr(node, 'input_activation_memory', 0)), ('second_input', getattr(node, 'second_input_activation_memory', 0)), ('weights', getattr(node, 'weight_memory', 0)), ('bias', getattr(node, 'bias_memory', 0)), ('constants', getattr(node, 'constants_memory', 0))):
        size_i = nonnegative_int(size)
        if size_i:
            transfers.append(DMATransferSpec(name=name, direction='L2_TO_L1', number_of_2d_copies=1, number_of_1d_copies=1, length_1d_copy=size_i, logical_bytes=size_i))
    output = nonnegative_int(getattr(node, 'output_activation_memory', 0))
    if output:
        transfers.append(DMATransferSpec(name='output', direction='L1_TO_L2', number_of_2d_copies=1, number_of_1d_copies=1, length_1d_copy=output, logical_bytes=output))
    return NodeExecutionSpec(dma_transfers=tuple(transfers))
