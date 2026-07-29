from __future__ import annotations

from typing import Any, List, Mapping

from ..config import AutoSpecConfig
from ..descriptors import AddKernelSpec, BankAccessPattern, DMATransferSpec, L1RegionSpec, NodeExecutionSpec
from ..utils import nonnegative_int


def make_add_kernel_spec(node: Any, kernel_name: str) -> AddKernelSpec:
    input_dims = list(getattr(node, 'input_dimensions', ()) or ())
    if len(input_dims) < 2:
        raise ValueError('add node must expose two-dimensional input_dimensions')
    channels = int(
        getattr(node, 'input_channels', 0)
        or getattr(node, 'output_channels', 0)
        or 1
    )
    return AddKernelSpec(
        name=kernel_name,
        height=int(input_dims[0]),
        width=int(input_dims[1]),
        channels=channels,
        input1_bits=int(getattr(node, 'input_activation_bits', 8) or 8),
        input2_bits=int(getattr(node, 'second_input_activation_bits', 8) or 8),
        output_bits=int(getattr(node, 'output_activation_bits', 8) or 8),
        out_requant=bool(getattr(node, 'out_requant_flag', True)),
    )


def make_add_execution_spec(
    node: Any,
    kernel_name: str,
    offsets: Mapping[str, int],
    config: AutoSpecConfig,
) -> NodeExecutionSpec:
    kernel = make_add_kernel_spec(node, kernel_name)
    input1_bytes = nonnegative_int(getattr(node, 'input_activation_memory', 0))
    input2_bytes = nonnegative_int(
        getattr(node, 'second_input_activation_memory', 0)
    )
    if input2_bytes <= 0:
        input2_bytes = input1_bytes
    output_bytes = nonnegative_int(getattr(node, 'output_activation_memory', 0))

    transfers: List[DMATransferSpec] = []
    for role, size in (('input', input1_bytes), ('second_input', input2_bytes)):
        if size:
            transfers.append(DMATransferSpec(
                name=role, direction='L2_TO_L1',
                number_of_2d_copies=1, number_of_1d_copies=1,
                length_1d_copy=size, logical_bytes=size,
            ))
    if output_bytes:
        transfers.append(DMATransferSpec(
            name='output', direction='L1_TO_L2',
            number_of_2d_copies=1, number_of_1d_copies=1,
            length_1d_copy=output_bytes, logical_bytes=output_bytes,
        ))

    regions = {
        'input': L1RegionSpec('input', int(offsets.get('input', 0)), input1_bytes),
        'second_input': L1RegionSpec(
            'second_input', int(offsets.get('second_input', 0)), input2_bytes
        ),
        'output': L1RegionSpec('output', int(offsets.get('output', 0)), output_bytes),
    }
    patterns = (
        BankAccessPattern(
            name='add_input1_reads', component='input_reads',
            correlation=config.add_input1_correlation,
            region_name='input', access_width_bytes=1, access_stride_bytes=4,
        ),
        BankAccessPattern(
            name='add_input2_reads', component='second_input_reads',
            correlation=config.add_input2_correlation,
            region_name='second_input', access_width_bytes=1, access_stride_bytes=4,
        ),
        BankAccessPattern(
            name='add_output_writes', component='output_writes',
            correlation=config.add_output_correlation,
            region_name='output', access_width_bytes=1, access_stride_bytes=4,
        ),
    )
    return NodeExecutionSpec(
        dma_transfers=tuple(transfers), compute_kernel=kernel,
        l1_regions=regions, bank_access_patterns=patterns,
        total_tiles=1, team_barriers_outside_kernel=4,
        team_barriers_inside_kernel=1,
        control_events={'dma_allocations': 1, 'dma_frees': 1, 'kernel_calls': 1},
    )
