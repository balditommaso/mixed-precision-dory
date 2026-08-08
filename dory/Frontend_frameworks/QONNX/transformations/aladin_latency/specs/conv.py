from __future__ import annotations

from math import ceil
from typing import Any, Dict, List, Mapping, Union

from ..config import AutoSpecConfig
from ..descriptors import BankAccessPattern, DMATransferSpec, KernelComputeSpec, L1RegionSpec, NodeExecutionSpec, PartitionStrategy
from ..utils import align_up, nonnegative_int


def infer_l1_offsets(node: Any, kernel: KernelComputeSpec, config: AutoSpecConfig) -> Dict[str, int]:
    """
    Produce one deterministic packed layout when source offsets are missing.

    This is a conservative approximation. The generated source parser should
    be preferred because bank conflicts depend on offsets modulo the bank
    interleaving period.
    """
    alignment = config.l1_alignment_bytes
    guard = config.l1_guard_bytes
    input_size = nonnegative_int(getattr(node, 'input_activation_memory', 0))
    second_input_size = nonnegative_int(
        getattr(node, 'second_input_activation_memory', 0)
    )
    output_size = nonnegative_int(getattr(node, 'output_activation_memory', 0))
    weight_size = nonnegative_int(getattr(node, 'weight_memory', 0))
    bias_size = nonnegative_int(getattr(node, 'bias_memory', 0))
    offsets: Dict[str, int] = {'input': 0}
    cursor = input_size
    if second_input_size:
        offsets['second_input'] = align_up(cursor + guard, alignment)
        cursor = offsets['second_input'] + second_input_size
    offsets['output'] = align_up(cursor + guard, alignment)
    cursor = offsets['output'] + output_size
    offsets['weights'] = align_up(cursor + guard, alignment)
    cursor = offsets['weights'] + weight_size
    if bias_size:
        offsets['bias'] = align_up(cursor + guard, alignment)
        cursor = offsets['bias'] + bias_size
    offsets['im2col'] = align_up(cursor, max(4, alignment))
    if kernel.is_depthwise:
        cursor = offsets['im2col'] + kernel.im2col_bytes_per_core * 8
        offsets['wt_buffer'] = align_up(cursor + guard, max(4, alignment))
    return offsets


def make_pulp_conv_kernel_spec(
    node: Any,
    kernel_name: str,
    *,
    flag_relu: bool = True,
    flag_batch_norm: bool = False,
) -> KernelComputeSpec:
    pads = list(getattr(node, 'pads', [0, 0, 0, 0]))
    while len(pads) < 4:
        pads.append(0)
    groups = int(getattr(node, 'group', 1) or 1)
    input_channels = int(node.input_channels)
    output_channels = int(node.output_channels)
    name_lower = kernel_name.lower()
    implementation = str(
        getattr(node, "implementation", "mac") or "mac"
    ).lower()
    is_depthwise = (
        'depthwise' in name_lower
        or (
            groups == input_channels
            and output_channels % max(1, input_channels) == 0
        )
    )
    kernel_kind = 'depthwise' if is_depthwise else ('grouped_conv' if groups > 1 else 'standard_conv')
    partition_strategy: PartitionStrategy = (
        'pulp_nn_depthwise_channels' if is_depthwise else 'pulp_nn_spatial'
    )
    return KernelComputeSpec(
        name=kernel_name,
        input_height=int(node.input_dimensions[0]),
        input_width=int(node.input_dimensions[1]),
        input_channels=input_channels,
        output_height=int(node.output_dimensions[0]),
        output_width=int(node.output_dimensions[1]),
        output_channels=output_channels,
        kernel_height=int(node.kernel_shape[0]),
        kernel_width=int(node.kernel_shape[1]),
        stride_height=int(node.strides[0]),
        stride_width=int(node.strides[1]),
        padding_top=int(pads[0]),
        implementation=implementation,
        padding_left=int(pads[1]),
        padding_bottom=int(pads[2]),
        padding_right=int(pads[3]),
        groups=groups,
        kernel_kind=kernel_kind,
        input_bits=int(node.input_activation_bits),
        weight_bits=int(node.weight_bits),
        output_bits=int(node.output_activation_bits),
        flag_relu=flag_relu,
        flag_batch_norm=flag_batch_norm,
        has_bias=nonnegative_int(getattr(node, 'bias_memory', 0)) > 0,
        partition_strategy=partition_strategy,
    )


def make_single_tile_dory_conv_spec(
    node: Any,
    kernel_name: str,
    *,
    l1_offsets: Union[Mapping[str, int], None] = None,
    flag_relu: bool = True,
    flag_batch_norm: bool = False,
    weight_correlation: float = 0.9,
    input_correlation: float = 0.2,
    im2col_correlation: float = 0.1,
    output_correlation: float = 0.1,
    bias_correlation: float = 0.8,
    lut_correlation: float = 0.75,
    lut_effective_banks: int = 4,
) -> NodeExecutionSpec:

    kernel = make_pulp_conv_kernel_spec(
        node,
        kernel_name,
        flag_relu=flag_relu,
        flag_batch_norm=flag_batch_norm,
    )

    is_lut = kernel.implementation == "lut"

    input_pixel_bytes = ceil(
        kernel.input_channels * kernel.input_bits / 8
    )
    output_pixel_bytes = ceil(
        kernel.output_channels * kernel.output_bits / 8
    )
    weight_bytes_per_position = ceil(
        kernel.input_channels * kernel.weight_bits / 8
    )
    weight_stride_output_channel = ceil(
        kernel.reduction_size * kernel.weight_bits / 8
    )

    bias_bytes = nonnegative_int(
        getattr(node, "bias_memory", 0)
    )

    l1 = (
        getattr(node, "tiling_dimensions", {})
        .get("L1", {})
    )
    lut_bytes = (
        nonnegative_int(l1.get("lut_memory", 0))
        if is_lut
        else 0
    )

    transfers: List[DMATransferSpec] = []

    if bias_bytes:
        transfers.append(
            DMATransferSpec(
                name="bias",
                direction="L2_TO_L1",
                number_of_2d_copies=1,
                number_of_1d_copies=1,
                length_1d_copy=bias_bytes,
                logical_bytes=bias_bytes,
            )
        )

    transfers.extend([
        DMATransferSpec(
            name="input",
            direction="L2_TO_L1",
            number_of_2d_copies=kernel.input_height,
            number_of_1d_copies=kernel.input_width,
            length_1d_copy=input_pixel_bytes,
            stride_2d=kernel.input_width * input_pixel_bytes,
            stride_1d=input_pixel_bytes,
            logical_bytes=nonnegative_int(
                node.input_activation_memory
            ),
        ),

        DMATransferSpec(
            name="weights",
            direction="L2_TO_L1",
            number_of_2d_copies=kernel.output_channels,
            number_of_1d_copies=(
                kernel.kernel_height * kernel.kernel_width
            ),
            length_1d_copy=weight_bytes_per_position,
            stride_2d=weight_stride_output_channel,
            stride_1d=weight_bytes_per_position,
            logical_bytes=nonnegative_int(
                node.weight_memory
            ),
            barrier_calls=2,
        ),
    ])

    if is_lut and lut_bytes:
        transfers.append(
            DMATransferSpec(
                name="lut",
                direction="L2_TO_L1",
                number_of_2d_copies=1,
                number_of_1d_copies=1,
                length_1d_copy=lut_bytes,
                logical_bytes=lut_bytes,
            )
        )

    transfers.append(
        DMATransferSpec(
            name="output",
            direction="L1_TO_L2",
            number_of_2d_copies=kernel.output_height,
            number_of_1d_copies=kernel.output_width,
            length_1d_copy=output_pixel_bytes,
            stride_2d=kernel.output_width * output_pixel_bytes,
            stride_1d=output_pixel_bytes,
            logical_bytes=nonnegative_int(
                node.output_activation_memory
            ),
        )
    )

    offsets = dict(l1_offsets or {})

    regions: Dict[str, L1RegionSpec] = {
        "input": L1RegionSpec(
            "input",
            offsets.get("input", 0),
            nonnegative_int(node.input_activation_memory),
        ),
        "output": L1RegionSpec(
            "output",
            offsets.get("output", 0),
            nonnegative_int(node.output_activation_memory),
        ),
        "weights": L1RegionSpec(
            "weights",
            offsets.get("weights", 0),
            nonnegative_int(node.weight_memory),
        ),
        "im2col": L1RegionSpec(
            "im2col",
            offsets.get("im2col", 0),
            kernel.im2col_bytes_per_core,
            per_core_stride_bytes=kernel.im2col_bytes_per_core,
        ),
    }

    if bias_bytes:
        regions["bias"] = L1RegionSpec(
            "bias",
            offsets.get("bias", 0),
            bias_bytes,
        )

    # NEW: LUT memory region.
    if is_lut and lut_bytes:
        regions["lut"] = L1RegionSpec(
            "lut",
            offsets.get("lut", 0),
            lut_bytes,
        )

    patterns: List[BankAccessPattern] = [
        BankAccessPattern(
            name="input_feature_map_reads",
            component="input_reads",
            correlation=input_correlation,
            region_name="input",
            access_width_bytes=4,
            access_stride_bytes=max(1, input_pixel_bytes),
        ),

        BankAccessPattern(
            name="private_im2col_writes",
            component="im2col_writes",
            correlation=im2col_correlation,
            region_name="im2col",
            access_width_bytes=4,
            access_stride_bytes=4,
        ),

        BankAccessPattern(
            name="private_im2col_reads",
            component="im2col_reads",
            correlation=im2col_correlation,
            region_name="im2col",
            access_width_bytes=4,
            access_stride_bytes=4,
        ),

        BankAccessPattern(
            name="shared_weight_reads",
            component="weight_reads",
            correlation=weight_correlation,
            broadcast_eligible=True,
            region_name="weights",
            access_width_bytes=4,
            access_stride_bytes=4,
        ),

        BankAccessPattern(
            name="output_writes",
            component="output_writes",
            correlation=output_correlation,
            region_name="output",
            access_width_bytes=1,
            access_stride_bytes=max(1, output_pixel_bytes),
        ),
    ]

    if bias_bytes:
        patterns.append(
            BankAccessPattern(
                name="shared_bias_reads",
                component="bias_reads",
                correlation=bias_correlation,
                broadcast_eligible=True,
                region_name="bias",
                access_width_bytes=4,
                access_stride_bytes=4,
            )
        )

    # NEW: this is the important LUT conflict model.
    if is_lut and lut_bytes:
        patterns.append(
            BankAccessPattern(
                name="lut_table_reads",
                component="lut_reads",
                requester_scope="active_cores",
                correlation=lut_correlation,
                region_name="lut",
                access_width_bytes=1,
                access_stride_bytes=1,
                effective_banks_override=lut_effective_banks,
                broadcast_eligible=False,
            )
        )

    return NodeExecutionSpec(
        dma_transfers=tuple(transfers),
        compute_kernel=kernel,
        l1_regions=regions,
        bank_access_patterns=tuple(patterns),
        total_tiles=1,
        team_barriers_outside_kernel=4,
        team_barriers_inside_kernel=1,
        control_events={
            "dma_allocations": 1,
            "dma_frees": 1,
            "kernel_calls": 1,
        },
    )


def make_single_tile_dory_depthwise_spec(
    node: Any,
    kernel_name: str,
    *,
    l1_offsets: Union[Mapping[str, int], None] = None,
    flag_relu: bool = True,
    flag_batch_norm: bool = False,
    weight_correlation: float = 0.10,
    input_correlation: float = 0.10,
    im2col_correlation: float = 0.05,
    output_correlation: float = 0.05,
    bias_correlation: float = 0.10,
) -> NodeExecutionSpec:
    kernel = make_pulp_conv_kernel_spec(
        node,
        kernel_name,
        flag_relu=flag_relu,
        flag_batch_norm=flag_batch_norm,
    )
    transfers: List[DMATransferSpec] = []
    for name, direction, size, barriers in (
        ('input', 'L2_TO_L1', getattr(node, 'input_activation_memory', 0), 1),
        ('weights', 'L2_TO_L1', getattr(node, 'weight_memory', 0), 2),
        ('bias', 'L2_TO_L1', getattr(node, 'bias_memory', 0), 1),
        ('output', 'L1_TO_L2', getattr(node, 'output_activation_memory', 0), 1),
    ):
        size_i = nonnegative_int(size)
        if size_i:
            transfers.append(
                DMATransferSpec(
                    name=name,
                    direction=direction,
                    number_of_2d_copies=1,
                    number_of_1d_copies=1,
                    length_1d_copy=size_i,
                    logical_bytes=size_i,
                    barrier_calls=barriers,
                )
            )

    offsets = dict(l1_offsets or {})
    im2col_stride = kernel.im2col_bytes_per_core
    wt_buffer_stride = 2 * kernel.kernel_height * kernel.kernel_width
    regions: Dict[str, L1RegionSpec] = {
        'input': L1RegionSpec(
            'input', offsets.get('input', 0),
            nonnegative_int(getattr(node, 'input_activation_memory', 0)),
        ),
        'output': L1RegionSpec(
            'output', offsets.get('output', 0),
            nonnegative_int(getattr(node, 'output_activation_memory', 0)),
        ),
        'weights': L1RegionSpec(
            'weights', offsets.get('weights', 0),
            nonnegative_int(getattr(node, 'weight_memory', 0)),
        ),
        'im2col': L1RegionSpec(
            'im2col', offsets.get('im2col', 0),
            im2col_stride,
            per_core_stride_bytes=im2col_stride,
        ),
        'wt_buffer': L1RegionSpec(
            'wt_buffer', offsets.get('wt_buffer', offsets.get('im2col', 0) + im2col_stride * 8),
            wt_buffer_stride,
            per_core_stride_bytes=wt_buffer_stride,
        ),
    }
    bias_bytes = nonnegative_int(getattr(node, 'bias_memory', 0))
    if bias_bytes:
        regions['bias'] = L1RegionSpec(
            'bias', offsets.get('bias', 0), bias_bytes
        )

    patterns: List[BankAccessPattern] = [
        BankAccessPattern(
            name='depthwise_input_reads',
            component='input_reads',
            correlation=input_correlation,
            region_name='input',
            access_width_bytes=4,
            access_stride_bytes=4,
        ),
        BankAccessPattern(
            name='depthwise_im2col_writes',
            component='im2col_writes',
            correlation=im2col_correlation,
            region_name='im2col',
            access_width_bytes=4,
            access_stride_bytes=4,
        ),
        BankAccessPattern(
            name='depthwise_im2col_reads',
            component='im2col_reads',
            correlation=im2col_correlation,
            region_name='im2col',
            access_width_bytes=4,
            access_stride_bytes=4,
        ),
        BankAccessPattern(
            name='depthwise_packed_weight_reads',
            component='weight_reads',
            correlation=weight_correlation,
            region_name='weights',
            access_width_bytes=1,
            access_stride_bytes=1,
        ),
        BankAccessPattern(
            name='depthwise_weight_unpack_writes',
            component='weight_unpack_writes',
            correlation=im2col_correlation,
            region_name='wt_buffer',
            access_width_bytes=1,
            access_stride_bytes=1,
        ),
        BankAccessPattern(
            name='depthwise_weight_buffer_reads',
            component='weight_buffer_reads',
            correlation=im2col_correlation,
            region_name='wt_buffer',
            access_width_bytes=4,
            access_stride_bytes=4,
        ),
        BankAccessPattern(
            name='depthwise_output_writes',
            component='output_writes',
            correlation=output_correlation,
            region_name='output',
            access_width_bytes=1,
            access_stride_bytes=max(1, kernel.output_channels),
        ),
    ]
    if bias_bytes:
        patterns.append(
            BankAccessPattern(
                name='depthwise_bias_reads',
                component='bias_reads',
                correlation=bias_correlation,
                region_name='bias',
                access_width_bytes=1,
                access_stride_bytes=1,
            )
        )
    return NodeExecutionSpec(
        dma_transfers=tuple(transfers),
        compute_kernel=kernel,
        l1_regions=regions,
        bank_access_patterns=tuple(patterns),
        total_tiles=1,
        team_barriers_outside_kernel=4,
        team_barriers_inside_kernel=1,
        control_events={
            'dma_allocations': 1,
            'dma_frees': 1,
            'kernel_calls': 1,
        },
    )
