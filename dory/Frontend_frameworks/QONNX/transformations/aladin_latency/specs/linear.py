from __future__ import annotations

from typing import Any, List, Mapping

from ..config import AutoSpecConfig
from ..descriptors import BankAccessPattern, DMATransferSpec, L1RegionSpec, LinearKernelSpec, NodeExecutionSpec
from ..utils import nonnegative_int


def make_linear_kernel_spec(node: Any, kernel_name: str) -> LinearKernelSpec:
    output_neurons = int(getattr(node, 'output_channels', 0) or 0)
    if output_neurons <= 0:
        output_dims = list(getattr(node, 'output_dimensions', ()) or ())
        if output_dims:
            output_neurons = 1
            for value in output_dims:
                output_neurons *= int(value)
    if output_neurons <= 0:
        output_bits = int(getattr(node, 'output_activation_bits', 8) or 8)
        output_neurons = max(
            1,
            nonnegative_int(getattr(node, 'output_activation_memory', 0)) * 8 // output_bits,
        )

    graph_macs = int(getattr(node, 'MACs', 0) or 0)
    if graph_macs > 0 and graph_macs % output_neurons == 0:
        input_features = graph_macs // output_neurons
    else:
        input_bits = int(getattr(node, 'input_activation_bits', 8) or 8)
        input_features = max(
            1,
            nonnegative_int(getattr(node, 'input_activation_memory', 0)) * 8 // input_bits,
        )

    constant_names = {
        str(name).lower() for name in getattr(node, 'constant_names', ()) or ()
    }
    return LinearKernelSpec(
        name=kernel_name,
        input_features=input_features,
        output_neurons=output_neurons,
        input_bits=int(getattr(node, 'input_activation_bits', 8) or 8),
        weight_bits=int(getattr(node, 'weight_bits', 8) or 8),
        output_bits=int(getattr(node, 'output_activation_bits', 8) or 8),
        flag_relu='relu' in str(getattr(node, 'op_type', '')).lower(),
        flag_batch_norm=bool({'kappa', 'lambda'} & constant_names),
        has_bias=nonnegative_int(getattr(node, 'bias_memory', 0)) > 0,
    )


def make_linear_execution_spec(
    node: Any,
    kernel_name: str,
    offsets: Mapping[str, int],
    config: AutoSpecConfig,
) -> NodeExecutionSpec:
    kernel = make_linear_kernel_spec(node, kernel_name)
    sizes = {
        'input': nonnegative_int(getattr(node, 'input_activation_memory', 0)),
        'weights': nonnegative_int(getattr(node, 'weight_memory', 0)),
        'bias': nonnegative_int(getattr(node, 'bias_memory', 0)),
        'output': nonnegative_int(getattr(node, 'output_activation_memory', 0)),
    }
    transfers: List[DMATransferSpec] = []
    for role in ('input', 'weights', 'bias'):
        if sizes[role]:
            transfers.append(DMATransferSpec(
                name=role, direction='L2_TO_L1',
                number_of_2d_copies=1, number_of_1d_copies=1,
                length_1d_copy=sizes[role], logical_bytes=sizes[role],
            ))
    if sizes['output']:
        transfers.append(DMATransferSpec(
            name='output', direction='L1_TO_L2',
            number_of_2d_copies=1, number_of_1d_copies=1,
            length_1d_copy=sizes['output'], logical_bytes=sizes['output'],
        ))
    regions = {
        role: L1RegionSpec(role, int(offsets.get(role, 0)), size)
        for role, size in sizes.items() if size > 0
    }
    patterns: List[BankAccessPattern] = [
        BankAccessPattern(
            name='linear_shared_input_reads', component='input_reads',
            correlation=config.linear_input_correlation,
            broadcast_eligible=True, region_name='input',
            access_width_bytes=4, access_stride_bytes=4,
        ),
        BankAccessPattern(
            name='linear_weight_reads', component='weight_reads',
            correlation=config.linear_weight_correlation,
            region_name='weights', access_width_bytes=4, access_stride_bytes=4,
        ),
        BankAccessPattern(
            name='linear_output_writes', component='output_writes',
            correlation=config.linear_output_correlation,
            region_name='output', access_width_bytes=1, access_stride_bytes=1,
        ),
    ]
    if sizes['bias']:
        patterns.append(BankAccessPattern(
            name='linear_bias_reads', component='bias_reads',
            correlation=config.linear_bias_correlation,
            region_name='bias', access_width_bytes=4, access_stride_bytes=4,
        ))
    return NodeExecutionSpec(
        dma_transfers=tuple(transfers), compute_kernel=kernel,
        l1_regions=regions, bank_access_patterns=tuple(patterns),
        total_tiles=1, team_barriers_outside_kernel=4,
        team_barriers_inside_kernel=1,
        control_events={'dma_allocations': 1, 'dma_frees': 1, 'kernel_calls': 1},
    )
