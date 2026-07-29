from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import Literal, Mapping, Tuple, Union

DMADirection = Literal['L2_TO_L1', 'L1_TO_L2']
PartitionMode = Literal['implementation_exact', 'ideal_balanced']
PartitionStrategy = Literal[
    'pulp_nn_spatial',
    'pulp_nn_depthwise_channels',
    'balanced_output_pixels',
    'single_core',
]
MemoryComponent = Literal[
    'input_reads',
    'second_input_reads',
    'im2col_writes',
    'im2col_reads',
    'weight_reads',
    'weight_unpack_writes',
    'weight_buffer_reads',
    'bias_reads',
    'output_writes',
]
RequesterScope = Literal['active_cores', 'all_cores', 'single', 'fixed']


class DMAKind(str, Enum):
    ONE_D = '1d'
    TWO_D = '2d'
    THREE_D = '3d'
    HWC_TO_CHW = 'hwc_to_chw'


@dataclass(frozen=True)
class NodeSourceMetadata:
    """Static metadata automatically extracted from one generated C file."""
    source_path: Union[str, None] = None
    kernel_name: Union[str, None] = None
    total_tiles: Union[int, None] = None
    l1_offsets: Mapping[str, int] = field(default_factory=dict)
    team_barriers_outside_kernel: Union[int, None] = None
    control_events: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DMATransferSpec:
    name: str
    direction: DMADirection
    number_of_2d_copies: int
    number_of_1d_copies: int
    length_1d_copy: int
    stride_2d: int = 0
    stride_1d: int = 0
    hwc_to_chw: bool = False
    logical_bytes: Union[int, None] = None
    submissions: int = 1
    barrier_calls: int = 1
    physical_bytes_override: Union[int, None] = None

    def __post_init__(self) -> None:
        for name in ('number_of_2d_copies', 'number_of_1d_copies', 'length_1d_copy', 'stride_2d', 'stride_1d', 'submissions', 'barrier_calls'):
            if getattr(self, name) < 0:
                raise ValueError(f'{self.name}: {name} cannot be negative')

    @property
    def geometry_bytes_per_submission(self) -> int:
        return self.number_of_2d_copies * self.number_of_1d_copies * self.length_1d_copy

    @property
    def geometry_bytes(self) -> int:
        return self.geometry_bytes_per_submission * self.submissions

    @property
    def physical_bytes(self) -> int:
        if self.physical_bytes_override is not None:
            return self.physical_bytes_override
        return self.geometry_bytes


@dataclass(frozen=True)
class L1RegionSpec:
    """L1 layout information extracted from generated layer code."""
    name: str
    base_offset: int
    size_bytes: int
    per_core_stride_bytes: int = 0

    def __post_init__(self) -> None:
        if self.base_offset < 0 or self.size_bytes < 0:
            raise ValueError('L1 region offsets and sizes cannot be negative')
        if self.per_core_stride_bytes < 0:
            raise ValueError('per_core_stride_bytes cannot be negative')


@dataclass(frozen=True)
class BankAccessPattern:
    """Map a memory-cycle component to an L1 bank-access pattern."""
    name: str
    component: MemoryComponent
    requester_scope: RequesterScope = 'active_cores'
    fixed_requesters: Union[int, None] = None
    correlation: float = 0.0
    broadcast_eligible: bool = False
    region_name: Union[str, None] = None
    effective_banks_override: Union[int, None] = None
    access_width_bytes: int = 4
    access_stride_bytes: int = 4
    concurrent_dma_requesters: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.correlation <= 1:
            raise ValueError('correlation must be in [0, 1]')
        if self.fixed_requesters is not None and self.fixed_requesters < 0:
            raise ValueError('fixed_requesters cannot be negative')
        if self.effective_banks_override is not None and self.effective_banks_override <= 0:
            raise ValueError('effective_banks_override must be positive')
        if self.access_width_bytes <= 0 or self.access_stride_bytes <= 0:
            raise ValueError('bank access width and stride must be positive')
        if self.concurrent_dma_requesters < 0:
            raise ValueError('concurrent_dma_requesters cannot be negative')


@dataclass(frozen=True)
class KernelComputeSpec:
    name: str
    input_height: int
    input_width: int
    input_channels: int
    output_height: int
    output_width: int
    output_channels: int
    kernel_height: int
    kernel_width: int
    stride_height: int
    stride_width: int
    padding_top: int
    padding_bottom: int
    padding_left: int
    padding_right: int
    groups: int = 1
    kernel_kind: str = 'standard_conv'
    input_bits: int = 8
    weight_bits: int = 4
    output_bits: int = 8
    flag_relu: bool = True
    flag_batch_norm: bool = False
    has_bias: bool = True
    partition_strategy: PartitionStrategy = 'pulp_nn_spatial'
    output_pixels_per_matmul: int = 2
    peak_mac_per_cycle_per_core: Union[float, None] = None

    def __post_init__(self) -> None:
        dimensions = (
            self.input_height,
            self.input_width,
            self.input_channels,
            self.output_height,
            self.output_width,
            self.output_channels,
            self.kernel_height,
            self.kernel_width,
            self.stride_height,
            self.stride_width,
            self.groups,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError('kernel dimensions, channels, strides, and groups must be positive')
        if self.input_channels % self.groups != 0:
            raise ValueError('input_channels must be divisible by groups')
        if self.output_pixels_per_matmul <= 0:
            raise ValueError('output_pixels_per_matmul must be positive')

    @property
    def is_depthwise(self) -> bool:
        return (
            self.kernel_kind == 'depthwise'
            or (
                self.groups == self.input_channels
                and self.output_channels % self.input_channels == 0
            )
        )

    @property
    def is_grouped(self) -> bool:
        return self.groups > 1 and not self.is_depthwise

    @property
    def channels_per_group(self) -> int:
        return self.input_channels // self.groups

    @property
    def reduction_size(self) -> int:
        return self.channels_per_group * self.kernel_height * self.kernel_width

    @property
    def macs_per_output_pixel(self) -> int:
        return self.output_channels * self.reduction_size

    @property
    def output_pixels(self) -> int:
        return self.output_height * self.output_width

    @property
    def total_macs(self) -> int:
        return self.output_pixels * self.macs_per_output_pixel

    @property
    def im2col_bytes_per_core(self) -> int:
        if self.is_depthwise:
            rows = self.input_height + self.padding_top + self.padding_bottom
            return 2 * (self.kernel_width * rows + self.kernel_width)
        return self.output_pixels_per_matmul * self.reduction_size


@dataclass(frozen=True)
class PoolKernelSpec:
    """Descriptor for the supplied two-pass PULP-NN max-pooling kernel."""
    name: str
    input_height: int
    input_width: int
    channels: int
    output_height: int
    output_width: int
    kernel_height: int
    kernel_width: int
    stride_height: int
    stride_width: int
    padding_top: int
    padding_bottom: int
    padding_left: int
    padding_right: int
    input_bits: int = 8
    output_bits: int = 8

    def __post_init__(self) -> None:
        values = (
            self.input_height, self.input_width, self.channels,
            self.output_height, self.output_width,
            self.kernel_height, self.kernel_width,
            self.stride_height, self.stride_width,
        )
        if any(value <= 0 for value in values):
            raise ValueError('pooling dimensions, channels, kernels, and strides must be positive')

    @property
    def output_values(self) -> int:
        return self.output_height * self.output_width * self.channels


@dataclass(frozen=True)
class LinearKernelSpec:
    """Descriptor for the supplied signed 8-bit PULP-NN linear kernel."""
    name: str
    input_features: int
    output_neurons: int
    input_bits: int = 8
    weight_bits: int = 8
    output_bits: int = 8
    flag_relu: bool = True
    flag_batch_norm: bool = False
    has_bias: bool = True

    def __post_init__(self) -> None:
        if self.input_features <= 0 or self.output_neurons <= 0:
            raise ValueError('linear input_features and output_neurons must be positive')

    @property
    def total_macs(self) -> int:
        return self.input_features * self.output_neurons


@dataclass(frozen=True)
class AddKernelSpec:
    """Descriptor for the supplied quantized signed 8-bit add kernel."""
    name: str
    height: int
    width: int
    channels: int
    input1_bits: int = 8
    input2_bits: int = 8
    output_bits: int = 8
    out_requant: bool = True

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0 or self.channels <= 0:
            raise ValueError('add tensor dimensions and channels must be positive')

    @property
    def total_values(self) -> int:
        return self.height * self.width * self.channels


@dataclass(frozen=True)
class NodeExecutionSpec:
    dma_transfers: Tuple[DMATransferSpec, ...]
    compute_kernel: Union[ComputeKernelSpec, None] = None
    l1_regions: Mapping[str, L1RegionSpec] = field(default_factory=dict)
    bank_access_patterns: Tuple[BankAccessPattern, ...] = ()
    total_tiles: int = 1
    team_barriers_outside_kernel: int = 0
    team_barriers_inside_kernel: int = 0
    control_events: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CoreWork:
    core_id: int
    output_y_start: int
    output_y_stop: int
    output_x_start: int
    output_x_stop: int
    output_pixels: int
    macs: int

    @property
    def active(self) -> bool:
        return self.output_pixels > 0


@dataclass(frozen=True)
class DepthwiseCoreWork:
    core_id: int
    start_pair: int
    stop_pair: int
    channel_pairs: int
    output_values: int
    macs: int

    @property
    def active(self) -> bool:
        return self.channel_pairs > 0


ComputeKernelSpec = Union[
    KernelComputeSpec,
    PoolKernelSpec,
    LinearKernelSpec,
    AddKernelSpec,
]
