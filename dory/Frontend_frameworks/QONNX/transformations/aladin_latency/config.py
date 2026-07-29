from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class ExecutionConfig:
    """Build/runtime configuration for one inference run."""
    num_cores: int
    single_core_dma: bool = False
    always_block_dma_transfers: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.num_cores, int) or not 1 <= self.num_cores <= 8:
            raise ValueError('num_cores must be an integer between 1 and 8')

    @property
    def is_power_of_two(self) -> bool:
        return self.num_cores & self.num_cores - 1 == 0


@dataclass(frozen=True)
class DMAHardwareModel:
    read_bandwidth_bytes_per_cycle: float
    write_bandwidth_bytes_per_cycle: float
    transaction_startup_cycles: int = 10
    push_cycles: int = 2
    barrier_call_cycles: int = 4
    bandwidth_efficiency: float = 0.9
    pessimistic_bandwidth_efficiency: float = 0.8
    pessimistic_startup_factor: float = 1.15

    def __post_init__(self) -> None:
        if self.read_bandwidth_bytes_per_cycle <= 0:
            raise ValueError('read DMA bandwidth must be positive')
        if self.write_bandwidth_bytes_per_cycle <= 0:
            raise ValueError('write DMA bandwidth must be positive')
        if self.transaction_startup_cycles < 0:
            raise ValueError('transaction_startup_cycles cannot be negative')
        if self.push_cycles < 0 or self.barrier_call_cycles < 0:
            raise ValueError('DMA software cycle costs cannot be negative')
        if not 0 < self.bandwidth_efficiency <= 1:
            raise ValueError('bandwidth_efficiency must be in (0, 1]')
        if not 0 < self.pessimistic_bandwidth_efficiency <= 1:
            raise ValueError('pessimistic_bandwidth_efficiency must be in (0, 1]')
        if self.pessimistic_startup_factor < 1:
            raise ValueError('pessimistic_startup_factor must be >= 1')


@dataclass(frozen=True)
class L1BankModel:
    bank_count: int = 16
    interleaving_bytes: int = 4
    accesses_per_bank_per_cycle: int = 1
    same_address_broadcast: Union[bool, None] = None
    dma_shares_bank_ports: bool = True
    arbitration: str = 'round_robin'

    def __post_init__(self) -> None:
        if self.bank_count <= 0:
            raise ValueError('bank_count must be positive')
        if self.interleaving_bytes <= 0:
            raise ValueError('interleaving_bytes must be positive')
        if self.accesses_per_bank_per_cycle <= 0:
            raise ValueError('accesses_per_bank_per_cycle must be positive')


@dataclass(frozen=True)
class PessimismConfig:
    """Margins used when exact microarchitectural behavior is unknown."""
    bank_correlation_inflation: float = 0.15
    bank_spread_haircut: float = 0.15
    conflict_excess_factor: float = 1.1
    kernel_safety_factor: float = 1.4
    depthwise_kernel_safety_factor: float = 1.75
    grouped_kernel_safety_factor: float = 1.50
    pooling_kernel_safety_factor: float = 1.40
    pooling_memory_pressure_factor: float = 1.25
    linear_kernel_safety_factor: float = 1.10
    add_kernel_safety_factor: float = 1.20
    add_memory_pressure_factor: float = 1.35
    dma_safety_factor: float = 1.1
    layer_fixed_overhead_cycles: int = 100
    kernel_launch_cycles: int = 20
    team_barrier_cycles: int = 25
    dma_allocate_cycles: int = 8
    dma_free_cycles: int = 8
    assume_broadcast_for_expected: bool = False
    assume_broadcast_for_pessimistic: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.bank_correlation_inflation <= 1:
            raise ValueError('bank_correlation_inflation must be in [0, 1]')
        if not 0 <= self.bank_spread_haircut < 1:
            raise ValueError('bank_spread_haircut must be in [0, 1)')
        if self.conflict_excess_factor < 1:
            raise ValueError('conflict_excess_factor must be >= 1')
        if (
            self.kernel_safety_factor < 1
            or self.depthwise_kernel_safety_factor < 1
            or self.grouped_kernel_safety_factor < 1
            or self.pooling_kernel_safety_factor < 1
            or self.pooling_memory_pressure_factor < 1
            or self.linear_kernel_safety_factor < 1
            or self.add_kernel_safety_factor < 1
            or self.add_memory_pressure_factor < 1
            or self.dma_safety_factor < 1
        ):
            raise ValueError('safety factors must be >= 1')


@dataclass(frozen=True)
class AutoSpecConfig:
    """
    Global rules used to build execution metadata for every graph node.

    No per-node execution-spec mapping is required. If ``generated_code_dir``
    is set, the factory automatically reads ``<node.name>.c`` or
    ``<node.prefixed_name>.c`` and extracts static metadata. Missing metadata is
    inferred conservatively from the HW_node fields.
    """
    generated_code_dir: Union[str, Path, None] = None
    source_suffix: str = '.c'
    parse_generated_code: bool = True
    l1_alignment_bytes: int = 8
    l1_guard_bytes: int = 8
    weight_correlation: float = 0.9
    bias_correlation: float = 0.8
    input_correlation: float = 0.2
    im2col_correlation: float = 0.1
    output_correlation: float = 0.1
    depthwise_weight_correlation: float = 0.10
    depthwise_input_correlation: float = 0.10
    depthwise_im2col_correlation: float = 0.05
    depthwise_output_correlation: float = 0.05
    pooling_input_correlation: float = 0.15
    pooling_output_correlation: float = 0.10
    linear_input_correlation: float = 0.55
    linear_weight_correlation: float = 0.35
    linear_bias_correlation: float = 0.30
    linear_output_correlation: float = 0.10
    add_input1_correlation: float = 0.25
    add_input2_correlation: float = 0.25
    add_output_correlation: float = 0.10
    default_team_barriers_outside_kernel: int = 4
    default_team_barriers_inside_pulp_conv: int = 1
    default_dma_allocations: int = 1
    default_dma_frees: int = 1
    default_kernel_calls: int = 1
    block_dma_and_compute: bool = True

    def __post_init__(self) -> None:
        if self.l1_alignment_bytes <= 0:
            raise ValueError('l1_alignment_bytes must be positive')
        if self.l1_guard_bytes < 0:
            raise ValueError('l1_guard_bytes cannot be negative')
        for name in (
            'weight_correlation',
            'bias_correlation',
            'input_correlation',
            'im2col_correlation',
            'output_correlation',
            'depthwise_weight_correlation',
            'depthwise_input_correlation',
            'depthwise_im2col_correlation',
            'depthwise_output_correlation',
            'pooling_input_correlation',
            'pooling_output_correlation',
            'linear_input_correlation',
            'linear_weight_correlation',
            'linear_bias_correlation',
            'linear_output_correlation',
            'add_input1_correlation',
            'add_input2_correlation',
            'add_output_correlation',
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1]')


@dataclass(frozen=True)
class KernelCostModel:
    """
    Source-level instruction-cost approximation.

    Defaults are intentionally conservative. They should later be replaced by
    instruction traces or calibrated values for the selected ISA and compiler.
    """
    dotp4_cycles: float = 1.0
    scalar_mac_cycles: float = 1.0
    input_vector_load_cycles: float = 1.0
    input_scalar_load_cycles: float = 1.0
    weight_unpack_load_cycles: float = 1.0
    weight_unpack_compute_cycles: float = 7.0
    weight_scalar_load_cycles: float = 1.0
    bias_load_cycles: float = 1.0
    quant_relu_cycles_per_output: float = 5.0
    output_store_cycles: float = 1.0
    im2col_input_read_cycles_per_byte: float = 1.0
    im2col_write_cycles_per_byte: float = 1.0
    im2col_zero_write_cycles_per_byte: float = 1.0
    im2col_copy_call_overhead_cycles: float = 5.0
    im2col_zero_call_overhead_cycles: float = 5.0
    vector_loop_overhead_cycles: float = 1.0
    tail_loop_overhead_cycles: float = 2.0
    output_channel_group_overhead_cycles: float = 8.0
    matmul_call_overhead_cycles: float = 12.0
    odd_pixel_channel_overhead_cycles: float = 3.0
    output_pixel_loop_overhead_cycles: float = 4.0

    # Two-pass max-pooling source model. Costs are per scalar channel value;
    # compare-and-replace is typically vectorized inside the helper routine.
    pool_copy_cycles_per_value: float = 1.50
    pool_compare_cycles_per_value: float = 1.75
    pool_window_setup_cycles: float = 8.0
    pool_row_setup_cycles: float = 10.0

    # Fully-connected / linear source model.
    linear_input_vector_load_cycles: float = 1.0
    linear_weight_vector_load_cycles: float = 1.0
    linear_input_scalar_load_cycles: float = 1.0
    linear_weight_scalar_load_cycles: float = 1.0
    linear_pair_loop_overhead_cycles: float = 1.0
    linear_tail_loop_overhead_cycles: float = 1.0
    linear_neuron_pair_setup_cycles: float = 14.0
    linear_single_neuron_setup_cycles: float = 10.0

    # Quantized elementwise-add source model.
    add_input_load_cycles_per_value: float = 1.0
    add_input_requant_cycles_per_operand: float = 3.0
    add_clip_cycles_per_value: float = 1.0
    add_sum_cycles_per_value: float = 1.0
    add_output_requant_cycles_per_value: float = 3.0
    add_output_store_cycles_per_value: float = 1.0
    add_group4_overhead_cycles: float = 6.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f'{name} cannot be negative')
