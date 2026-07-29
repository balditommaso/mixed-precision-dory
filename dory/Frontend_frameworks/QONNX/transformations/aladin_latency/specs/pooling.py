from __future__ import annotations

from typing import Any, List, Mapping

from ..config import AutoSpecConfig
from ..descriptors import BankAccessPattern, DMATransferSpec, L1RegionSpec, NodeExecutionSpec, PoolKernelSpec
from ..utils import nonnegative_int


def make_pool_kernel_spec(node: Any, kernel_name: str) -> PoolKernelSpec:
    pads = list(getattr(node, 'pads', [0, 0, 0, 0]) or [0, 0, 0, 0])
    while len(pads) < 4:
        pads.append(0)
    channels = int(
        getattr(node, 'input_channels', 0)
        or getattr(node, 'output_channels', 0)
        or 1
    )
    return PoolKernelSpec(
        name=kernel_name,
        input_height=int(node.input_dimensions[0]),
        input_width=int(node.input_dimensions[1]),
        channels=channels,
        output_height=int(node.output_dimensions[0]),
        output_width=int(node.output_dimensions[1]),
        kernel_height=int(node.kernel_shape[0]),
        kernel_width=int(node.kernel_shape[1]),
        stride_height=int(node.strides[0]),
        stride_width=int(node.strides[1]),
        padding_top=int(pads[0]),
        padding_left=int(pads[1]),
        padding_bottom=int(pads[2]),
        padding_right=int(pads[3]),
        input_bits=int(getattr(node, 'input_activation_bits', 8) or 8),
        output_bits=int(getattr(node, 'output_activation_bits', 8) or 8),
    )


def make_pool_execution_spec(
    node: Any,
    kernel_name: str,
    offsets: Mapping[str, int],
    config: AutoSpecConfig,
) -> NodeExecutionSpec:
    kernel = make_pool_kernel_spec(node, kernel_name)
    input_bytes = nonnegative_int(getattr(node, 'input_activation_memory', 0))
    output_bytes = nonnegative_int(getattr(node, 'output_activation_memory', 0))
    transfers: List[DMATransferSpec] = []
    if input_bytes:
        transfers.append(DMATransferSpec(
            name='input', direction='L2_TO_L1',
            number_of_2d_copies=1, number_of_1d_copies=1,
            length_1d_copy=input_bytes, logical_bytes=input_bytes,
        ))
    if output_bytes:
        transfers.append(DMATransferSpec(
            name='output', direction='L1_TO_L2',
            number_of_2d_copies=1, number_of_1d_copies=1,
            length_1d_copy=output_bytes, logical_bytes=output_bytes,
        ))
    regions = {
        'input': L1RegionSpec('input', int(offsets.get('input', 0)), input_bytes),
        'output': L1RegionSpec('output', int(offsets.get('output', 0)), output_bytes),
    }
    patterns = (
        BankAccessPattern(
            name='pool_input_and_inplace_reads', component='input_reads',
            correlation=config.pooling_input_correlation,
            region_name='input', access_width_bytes=4, access_stride_bytes=4,
        ),
        BankAccessPattern(
            name='pool_inplace_and_output_writes', component='output_writes',
            correlation=config.pooling_output_correlation,
            region_name='output', access_width_bytes=4, access_stride_bytes=4,
        ),
    )
    return NodeExecutionSpec(
        dma_transfers=tuple(transfers), compute_kernel=kernel,
        l1_regions=regions, bank_access_patterns=patterns,
        total_tiles=1, team_barriers_outside_kernel=4,
        team_barriers_inside_kernel=2,
        control_events={'dma_allocations': 1, 'dma_frees': 1, 'kernel_calls': 1},
    )
