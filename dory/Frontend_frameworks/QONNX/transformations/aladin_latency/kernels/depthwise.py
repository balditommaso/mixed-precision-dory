from __future__ import annotations

from math import ceil, log2
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..banks import apply_bank_penalties
from ..config import ExecutionConfig, KernelCostModel, L1BankModel, PessimismConfig
from ..descriptors import BankAccessPattern, DepthwiseCoreWork, KernelComputeSpec, L1RegionSpec
from ..hardware import get_peak_mac_per_cycle_per_core
from ..utils import floor_log2


def partition_pulp_nn_depthwise_channels(
    kernel: KernelComputeSpec,
    num_cores: int,
) -> List[DepthwiseCoreWork]:
    """Mirror the depthwise kernel's channel-pair partitioning."""
    pair_count = ceil(kernel.output_channels / 2)
    if pair_count <= 0:
        return []
    log2_cores = floor_log2(num_cores)
    chunk = (pair_count >> log2_cores) + int(
        (pair_count & (num_cores - 1)) != 0
    )
    result: List[DepthwiseCoreWork] = []
    kernel_size = kernel.kernel_height * kernel.kernel_width
    outputs_per_pair = kernel.output_height * kernel.output_width * 2
    macs_per_pair = outputs_per_pair * kernel_size
    for core_id in range(num_cores):
        start_pair = min(chunk * core_id, pair_count)
        stop_pair = min(start_pair + chunk, pair_count)
        pairs = max(0, stop_pair - start_pair)
        output_values = pairs * outputs_per_pair
        result.append(
            DepthwiseCoreWork(
                core_id=core_id,
                start_pair=start_pair,
                stop_pair=stop_pair,
                channel_pairs=pairs,
                output_values=output_values,
                macs=pairs * macs_per_pair,
            )
        )
    return result


def _depthwise_horizontal_zero_elements(kernel: KernelComputeSpec) -> int:
    """Approximate scalar horizontal-padding writes for one input channel."""
    invalid = 0
    for out_x in range(kernel.output_width):
        input_x0 = out_x * kernel.stride_width - kernel.padding_left
        for ker_x in range(kernel.kernel_width):
            x = input_x0 + ker_x
            if x < 0 or x >= kernel.input_width:
                invalid += kernel.input_height
    return invalid


def depthwise_components_for_core(
    kernel: KernelComputeSpec,
    work: DepthwiseCoreWork,
    cost: KernelCostModel,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    pairs = work.channel_pairs
    kernel_size = kernel.kernel_height * kernel.kernel_width
    vector_columns = kernel_size // 4
    scalar_columns = kernel_size % 4
    padded_kernel_x_vectors = ceil(kernel.kernel_width / 4)
    padded_rows = kernel.input_height + kernel.padding_top + kernel.padding_bottom
    output_positions_per_pair = kernel.output_height * kernel.output_width

    dotp4 = pairs * output_positions_per_pair * 2 * vector_columns
    scalar_macs = pairs * output_positions_per_pair * 2 * scalar_columns
    packed_weight_loads = pairs * kernel_size
    unpacked_weight_stores = pairs * kernel_size * 2
    weight_buffer_vector_reads = dotp4
    weight_buffer_scalar_reads = scalar_macs

    # For each output x, the supplied kernel builds two private im2col strips,
    # one for each channel in the channel pair.
    im2col_input_vector_loads = (
        pairs
        * kernel.output_width
        * kernel.input_height
        * padded_kernel_x_vectors
        * 2
    )
    im2col_vector_writes = (
        pairs
        * kernel.output_width
        * padded_rows
        * padded_kernel_x_vectors
        * 2
    )
    im2col_horizontal_zero_writes = (
        pairs * 2 * _depthwise_horizontal_zero_elements(kernel)
    )
    im2col_vector_reads = dotp4
    im2col_scalar_reads = scalar_macs

    output_values = min(
        work.output_values,
        pairs * output_positions_per_pair * 2,
    )
    bias_loads = output_values * int(kernel.has_bias)
    quantized_outputs = output_values
    output_stores = output_values

    counts: Dict[str, int] = {
        'channel_pairs': pairs,
        'dotp4': dotp4,
        'scalar_macs': scalar_macs,
        'packed_weight_loads': packed_weight_loads,
        'unpacked_weight_stores': unpacked_weight_stores,
        'weight_buffer_vector_reads': weight_buffer_vector_reads,
        'weight_buffer_scalar_reads': weight_buffer_scalar_reads,
        'im2col_input_vector_loads': im2col_input_vector_loads,
        'im2col_vector_writes': im2col_vector_writes,
        'im2col_horizontal_zero_writes': im2col_horizontal_zero_writes,
        'im2col_vector_reads': im2col_vector_reads,
        'im2col_scalar_reads': im2col_scalar_reads,
        'bias_loads': bias_loads,
        'quantized_outputs': quantized_outputs,
        'output_stores': output_stores,
    }

    arithmetic = dotp4 * cost.dotp4_cycles + scalar_macs * cost.scalar_mac_cycles
    unpack_compute = packed_weight_loads * 2.0
    quantization = quantized_outputs * cost.quant_relu_cycles_per_output
    input_reads = im2col_input_vector_loads * cost.input_vector_load_cycles
    im2col_writes = (
        im2col_vector_writes * cost.im2col_write_cycles_per_byte
        + im2col_horizontal_zero_writes * cost.im2col_zero_write_cycles_per_byte
    )
    im2col_reads = (
        im2col_vector_reads * cost.input_vector_load_cycles
        + im2col_scalar_reads * cost.input_scalar_load_cycles
    )
    weight_reads = packed_weight_loads * cost.weight_unpack_load_cycles
    weight_unpack_writes = unpacked_weight_stores * cost.output_store_cycles
    weight_buffer_reads = (
        weight_buffer_vector_reads * cost.input_vector_load_cycles
        + weight_buffer_scalar_reads * cost.input_scalar_load_cycles
    )
    bias_reads = bias_loads * cost.bias_load_cycles
    output_writes = output_stores * cost.output_store_cycles

    control = (
        pairs * 12.0
        + pairs * kernel.output_width * 8.0
        + output_values * cost.output_pixel_loop_overhead_cycles
        + dotp4 * 0.5 * cost.vector_loop_overhead_cycles
        + scalar_macs * 0.5 * cost.tail_loop_overhead_cycles
    )

    components: Dict[str, float] = {
        'arithmetic': arithmetic,
        'unpack_compute': unpack_compute,
        'quantization': quantization,
        'control': control,
        'input_reads': input_reads,
        'im2col_writes': im2col_writes,
        'im2col_reads': im2col_reads,
        'weight_reads': weight_reads,
        'weight_unpack_writes': weight_unpack_writes,
        'weight_buffer_reads': weight_buffer_reads,
        'bias_reads': bias_reads,
        'output_writes': output_writes,
    }
    components['base_total'] = sum(components.values())
    return counts, components


def estimate_depthwise_compute(
    kernel: KernelComputeSpec,
    hw_spec: Mapping[str, Any],
    execution: ExecutionConfig,
    cost: KernelCostModel,
    bank_model: L1BankModel,
    patterns: Sequence[BankAccessPattern],
    regions: Mapping[str, L1RegionSpec],
    pessimism: PessimismConfig,
    peak_key: str,
    calibration_scale: float = 1.0,
) -> Dict[str, Any]:
    work = partition_pulp_nn_depthwise_channels(kernel, execution.num_cores)
    active_cores = sum(item.active for item in work)
    peak_per_core = get_peak_mac_per_cycle_per_core(hw_spec, peak_key, kernel)
    core_results: List[Dict[str, Any]] = []
    for item in work:
        if not item.active:
            core_results.append({
                'core_id': item.core_id,
                'active': False,
                'channel_pairs': 0,
                'macs': 0,
                'mac_lower_bound_cycles': 0,
                'operation_counts': {},
                'base_components': {},
                'bank_model': {},
                'expected_cycles': 0,
                'pessimistic_cycles': 0,
            })
            continue
        counts, components = depthwise_components_for_core(kernel, item, cost)
        bank_result = apply_bank_penalties(
            components,
            patterns,
            regions,
            bank_model,
            execution,
            active_cores,
            pessimism,
        )
        # The supplied depthwise kernel rebuilds channel-pair im2col strips for
        # every output x. Strided layers pay additional pointer, boundary, and
        # staging overhead that is not represented by the pure operation counts.
        expected_stride_factor = (
            1.0
            + 0.05 * max(0, kernel.stride_height - 1)
            + 0.05 * max(0, kernel.stride_width - 1)
        )
        pessimistic_stride_factor = (
            1.0
            + 0.10 * max(0, kernel.stride_height - 1)
            + 0.10 * max(0, kernel.stride_width - 1)
        )
        expected_cycles = ceil(
            bank_result['cycles_expected_before_safety']
            * expected_stride_factor
            * calibration_scale
        )
        pessimistic_cycles = ceil(
            bank_result['cycles_pessimistic_before_safety']
            * pessimism.depthwise_kernel_safety_factor
            * pessimistic_stride_factor
            * calibration_scale
        )
        core_results.append({
            'core_id': item.core_id,
            'active': True,
            'channel_pair_range': (item.start_pair, item.stop_pair),
            'channel_pairs': item.channel_pairs,
            'output_values': item.output_values,
            'macs': item.macs,
            'mac_lower_bound_cycles': ceil(item.macs / peak_per_core),
            'operation_counts': counts,
            'base_components': components,
            'bank_model': bank_result,
            'expected_cycles': expected_cycles,
            'pessimistic_cycles': pessimistic_cycles,
        })
    critical_expected = max(core_results, key=lambda item: item['expected_cycles'])
    critical_pessimistic = max(core_results, key=lambda item: item['pessimistic_cycles'])
    return {
        'model': 'depthwise_source_operations_plus_probabilistic_banks',
        'kernel_name': kernel.name,
        'kernel_kind': kernel.kernel_kind,
        'requested_cores': execution.num_cores,
        'active_cores': active_cores,
        'peak_mac_per_cycle_per_core': peak_per_core,
        'mac_lower_bound_cycles': max(item['mac_lower_bound_cycles'] for item in core_results),
        'expected_cycles': critical_expected['expected_cycles'],
        'pessimistic_cycles': critical_pessimistic['pessimistic_cycles'],
        'critical_core_expected': critical_expected['core_id'],
        'critical_core_pessimistic': critical_pessimistic['core_id'],
        'core_results': core_results,
        'calibration_scale': calibration_scale,
    }
