from __future__ import annotations

from math import ceil
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..banks import apply_bank_penalties
from ..config import ExecutionConfig, KernelCostModel, L1BankModel, PessimismConfig
from ..descriptors import BankAccessPattern, CoreWork, KernelComputeSpec, L1RegionSpec, PartitionMode
from ..hardware import get_peak_mac_per_cycle_per_core
from ..utils import floor_log2
from .depthwise import estimate_depthwise_compute


def partition_pulp_nn_spatial(kernel: KernelComputeSpec, num_cores: int) -> List[CoreWork]:
    """Mirror the supplied PULP-NN convolution partitioning."""
    out_h = kernel.output_height
    out_w = kernel.output_width
    extra_chunk = out_h & num_cores - 1 != 0
    split_width = extra_chunk and out_w > 1 and (num_cores > 1)
    result: List[CoreWork] = []
    for core_id in range(num_cores):
        if split_width:
            reduced_cores = num_cores >> 1
            log2_cores = floor_log2(reduced_cores)
            reduced_core_id = core_id >> 1
            section = core_id & 1
            reduced_width = out_w >> 1
            odd_width = out_w & 1
            extra_reduced = out_h & reduced_cores - 1 != 0
            x_start = section * reduced_width
            x_stop = reduced_width + section * (reduced_width + odd_width)
        else:
            log2_cores = floor_log2(num_cores)
            reduced_core_id = core_id
            extra_reduced = extra_chunk
            x_start = 0
            x_stop = out_w
        chunk = (out_h >> log2_cores) + int(extra_reduced)
        y_start = min(chunk * reduced_core_id, out_h)
        y_stop = min(y_start + chunk, out_h)
        pixels = max(0, y_stop - y_start) * max(0, x_stop - x_start)
        result.append(CoreWork(core_id=core_id, output_y_start=y_start, output_y_stop=y_stop, output_x_start=x_start, output_x_stop=x_stop, output_pixels=pixels, macs=pixels * kernel.macs_per_output_pixel))
    return result


def partition_balanced_output_pixels(kernel: KernelComputeSpec, num_cores: int) -> List[CoreWork]:
    base = kernel.output_pixels // num_cores
    remainder = kernel.output_pixels % num_cores
    result: List[CoreWork] = []
    cursor = 0
    for core_id in range(num_cores):
        pixels = base + int(core_id < remainder)
        result.append(
            CoreWork(
                core_id=core_id, 
                output_y_start=0, 
                output_y_stop=0, 
                output_x_start=cursor, 
                output_x_stop=cursor + pixels, 
                output_pixels=pixels, 
                macs=pixels * kernel.macs_per_output_pixel
            )
        )
        cursor += pixels
    return result


def partition_kernel_work(kernel: KernelComputeSpec, num_cores: int, mode: PartitionMode) -> List[CoreWork]:
    if mode == 'ideal_balanced':
        return partition_balanced_output_pixels(kernel, num_cores)
    if kernel.partition_strategy == 'pulp_nn_spatial':
        return partition_pulp_nn_spatial(kernel, num_cores)
    if kernel.partition_strategy == 'balanced_output_pixels':
        return partition_balanced_output_pixels(kernel, num_cores)
    if kernel.partition_strategy == 'single_core':
        return [
            CoreWork(
                core_id=core_id, 
                output_y_start=0, 
                output_y_stop=kernel.output_height, 
                output_x_start=0, 
                output_x_stop=kernel.output_width, 
                output_pixels=kernel.output_pixels if core_id == 0 else 0, 
                macs=kernel.total_macs if core_id == 0 else 0
            ) for core_id in range(num_cores)
        ]
    raise ValueError(f'unsupported partition strategy: {kernel.partition_strategy}')


def _im2col_counts_for_core(kernel: KernelComputeSpec, work: CoreWork) -> Dict[str, int]:
    """Count valid/padded im2col bytes and helper calls for one core."""
    valid_bytes = 0
    zero_bytes = 0
    copy_calls = 0
    zero_calls = 0
    for out_y in range(work.output_y_start, work.output_y_stop):
        for out_x in range(work.output_x_start, work.output_x_stop):
            for ker_y in range(kernel.kernel_height):
                in_y = out_y * kernel.stride_height - kernel.padding_top + ker_y
                row_validity: List[bool] = []
                for ker_x in range(kernel.kernel_width):
                    in_x = out_x * kernel.stride_width - kernel.padding_left + ker_x
                    row_validity.append(
                        0 <= in_y < kernel.input_height and 0 <= in_x < kernel.input_width
                    )
                if all(row_validity):
                    valid_bytes += kernel.input_channels * kernel.kernel_width
                    copy_calls += 1
                else:
                    for valid in row_validity:
                        if valid:
                            valid_bytes += kernel.input_channels
                            copy_calls += 1
                        else:
                            zero_bytes += kernel.input_channels
                            zero_calls += 1
    return {
        'valid_bytes': valid_bytes, 
        'zero_bytes': zero_bytes, 
        'copy_calls': copy_calls, 
        'zero_calls': zero_calls
    }


def count_pulp_nn_operations_for_core(kernel: KernelComputeSpec, work: CoreWork) -> Dict[str, int]:
    """
    Count the operations executed by the supplied two-column matmul family.

    The tail is rounded to pairs because int4 values are packed two per byte.
    This is deliberately conservative for odd reduction sizes.
    """
    pixels = work.output_pixels
    pair_calls = pixels // kernel.output_pixels_per_matmul
    odd_pixels = pixels % kernel.output_pixels_per_matmul
    reduction = kernel.reduction_size
    vector_chunks = reduction // 8
    tail_steps = ceil(reduction % 8 / 2)
    groups4 = kernel.output_channels // 4
    tail_channels = kernel.output_channels % 4
    counts = {
        'pair_calls': pair_calls, 
        'odd_pixels': odd_pixels, 
        'odd_pixel_channels': odd_pixels * kernel.output_channels, 
        'vector_chunks': 0, 
        'tail_iterations': 0, 
        'dotp4': 0, 
        'scalar_macs': 0, 
        'input_vector_loads': 0, 
        'input_scalar_loads': 0, 
        'weight_unpack_calls': 0, 
        'weight_scalar_loads': 0, 
        'bias_loads': 0, 
        'quantized_outputs': 0, 
        'output_stores': 0, 
        'output_channel_groups': 0
    }
    if pair_calls:
        counts['output_channel_groups'] += pair_calls * groups4
        counts['vector_chunks'] += pair_calls * groups4 * vector_chunks
        counts['tail_iterations'] += pair_calls * groups4 * tail_steps
        counts['input_vector_loads'] += pair_calls * groups4 * vector_chunks * 4
        counts['weight_unpack_calls'] += pair_calls * groups4 * vector_chunks * 4
        counts['dotp4'] += pair_calls * groups4 * vector_chunks * 16
        counts['input_scalar_loads'] += pair_calls * groups4 * tail_steps * 4
        counts['weight_scalar_loads'] += pair_calls * groups4 * tail_steps * 4
        counts['scalar_macs'] += pair_calls * groups4 * tail_steps * 16
        counts['bias_loads'] += pair_calls * groups4 * 4 * int(kernel.has_bias)
        counts['quantized_outputs'] += pair_calls * groups4 * 8
        counts['output_stores'] += pair_calls * groups4 * 8
        counts['output_channel_groups'] += pair_calls * tail_channels
        counts['vector_chunks'] += pair_calls * tail_channels * vector_chunks
        counts['tail_iterations'] += pair_calls * tail_channels * tail_steps
        counts['input_vector_loads'] += pair_calls * tail_channels * vector_chunks * 4
        counts['weight_unpack_calls'] += pair_calls * tail_channels * vector_chunks
        counts['dotp4'] += pair_calls * tail_channels * vector_chunks * 4
        counts['input_scalar_loads'] += pair_calls * tail_channels * tail_steps * 4
        counts['weight_scalar_loads'] += pair_calls * tail_channels * tail_steps
        counts['scalar_macs'] += pair_calls * tail_channels * tail_steps * 4
        counts['bias_loads'] += pair_calls * tail_channels * int(kernel.has_bias)
        counts['quantized_outputs'] += pair_calls * tail_channels * 2
        counts['output_stores'] += pair_calls * tail_channels * 2
    if odd_pixels:
        channels = odd_pixels * kernel.output_channels
        counts['output_channel_groups'] += channels
        counts['vector_chunks'] += channels * vector_chunks
        counts['tail_iterations'] += channels * tail_steps
        counts['input_vector_loads'] += channels * vector_chunks * 2
        counts['weight_unpack_calls'] += channels * vector_chunks
        counts['dotp4'] += channels * vector_chunks * 2
        counts['input_scalar_loads'] += channels * tail_steps * 2
        counts['weight_scalar_loads'] += channels * tail_steps
        counts['scalar_macs'] += channels * tail_steps * 2
        counts['bias_loads'] += channels * int(kernel.has_bias)
        counts['quantized_outputs'] += channels
        counts['output_stores'] += channels
    counts.update(_im2col_counts_for_core(kernel, work))
    return counts


def operation_counts_to_cycles(counts: Mapping[str, int], cost: KernelCostModel, output_pixels: int) -> Dict[str, float]:
    """Convert source-level operation counts into cycle components."""
    arithmetic = counts['dotp4'] * cost.dotp4_cycles + counts['scalar_macs'] * cost.scalar_mac_cycles
    input_reads = counts['input_vector_loads'] * cost.input_vector_load_cycles + counts['input_scalar_loads'] * cost.input_scalar_load_cycles + counts['valid_bytes'] * cost.im2col_input_read_cycles_per_byte
    im2col_writes = counts['valid_bytes'] * cost.im2col_write_cycles_per_byte + counts['zero_bytes'] * cost.im2col_zero_write_cycles_per_byte
    im2col_reads = counts['input_vector_loads'] * cost.input_vector_load_cycles + counts['input_scalar_loads'] * cost.input_scalar_load_cycles
    weight_reads = counts['weight_unpack_calls'] * cost.weight_unpack_load_cycles + counts['weight_scalar_loads'] * cost.weight_scalar_load_cycles
    unpack_compute = counts['weight_unpack_calls'] * cost.weight_unpack_compute_cycles
    bias_reads = counts['bias_loads'] * cost.bias_load_cycles
    quantization = counts['quantized_outputs'] * cost.quant_relu_cycles_per_output
    output_writes = counts['output_stores'] * cost.output_store_cycles
    control = counts['copy_calls'] * cost.im2col_copy_call_overhead_cycles + counts['zero_calls'] * cost.im2col_zero_call_overhead_cycles + counts['vector_chunks'] * cost.vector_loop_overhead_cycles + counts['tail_iterations'] * cost.tail_loop_overhead_cycles + counts['output_channel_groups'] * cost.output_channel_group_overhead_cycles + counts['pair_calls'] * cost.matmul_call_overhead_cycles + counts['odd_pixel_channels'] * cost.odd_pixel_channel_overhead_cycles + output_pixels * cost.output_pixel_loop_overhead_cycles
    input_generation_reads = counts['valid_bytes'] * cost.im2col_input_read_cycles_per_byte
    components = {'arithmetic': arithmetic, 'unpack_compute': unpack_compute, 'quantization': quantization, 'control': control, 'input_reads': input_generation_reads, 'im2col_writes': im2col_writes, 'im2col_reads': im2col_reads, 'weight_reads': weight_reads, 'bias_reads': bias_reads, 'output_writes': output_writes}
    components['base_total'] = sum(components.values())
    return components


def estimate_grouped_compute_fallback(
    kernel: KernelComputeSpec,
    graph_macs: int,
    hw_spec: Mapping[str, Any],
    execution: ExecutionConfig,
    pessimism: PessimismConfig,
    peak_key: str,
    calibration_scale: float = 1.0,
) -> Dict[str, Any]:
    cluster_peak = float(hw_spec['peak MAC/cycle'][peak_key])
    active_cores = min(execution.num_cores, max(1, kernel.output_channels))
    scaled_peak = cluster_peak * active_cores / max(1, execution.num_cores)
    mac_lower = ceil(graph_macs / max(1e-9, scaled_peak))
    expected_efficiency = 0.45
    pessimistic_efficiency = 0.30
    expected = ceil(
        graph_macs / max(1e-9, scaled_peak * expected_efficiency)
        * calibration_scale
    )
    pessimistic = ceil(
        graph_macs / max(1e-9, scaled_peak * pessimistic_efficiency)
        * pessimism.grouped_kernel_safety_factor
        * calibration_scale
    )
    return {
        'model': 'group_aware_mac_fallback',
        'kernel_name': kernel.name,
        'kernel_kind': kernel.kernel_kind,
        'requested_cores': execution.num_cores,
        'active_cores': active_cores,
        'mac_lower_bound_cycles': mac_lower,
        'expected_cycles': expected,
        'pessimistic_cycles': pessimistic,
        'core_results': [],
        'calibration_scale': calibration_scale,
    }


def estimate_compute(
    kernel: KernelComputeSpec,
    hw_spec: Mapping[str, Any],
    execution: ExecutionConfig,
    cost: KernelCostModel,
    bank_model: L1BankModel,
    patterns: Sequence[BankAccessPattern],
    regions: Mapping[str, L1RegionSpec],
    pessimism: PessimismConfig,
    peak_key: str,
    partition_mode: PartitionMode,
    calibration_scale: float = 1.0,
) -> Dict[str, Any]:
    if kernel.is_depthwise:
        return estimate_depthwise_compute(
            kernel,
            hw_spec,
            execution,
            cost,
            bank_model,
            patterns,
            regions,
            pessimism,
            peak_key,
            calibration_scale,
        )

    if kernel.is_grouped:
        return estimate_grouped_compute_fallback(
            kernel,
            kernel.total_macs,
            hw_spec,
            execution,
            pessimism,
            peak_key,
            calibration_scale,
        )

    implementation = str(
        getattr(kernel, "implementation", "") or ""
    ).lower()

    is_lut = implementation == "lut"
    work = partition_kernel_work(
        kernel,
        execution.num_cores,
        partition_mode,
    )

    active_cores = sum(item.active for item in work)

    peak_per_core = (
        get_peak_mac_per_cycle_per_core(
            hw_spec,
            peak_key,
            kernel,
        )
    )

    core_results = []

    for item in work:
        if not item.active:
            core_results.append(
                {
                    "core_id": item.core_id,
                    "active": False,
                    "output_pixels": 0,
                    "macs": 0,
                    "lut_lookups": 0,
                    "mac_lower_bound_cycles": 0,
                    "operation_counts": {},
                    "base_components": {},
                    "bank_model": {},
                    "expected_cycles": 0,
                    "pessimistic_cycles": 0,
                }
            )
            continue

        counts = (
            count_pulp_nn_operations_for_core(kernel, item)
        )

        components = operation_counts_to_cycles(
            counts,
            cost,
            item.output_pixels,
        )

        lookup_ops = 0

        if is_lut:
            lookup_ops = int(item.macs)

            components["arithmetic"] = (lookup_ops * cost.lut_accumulate_cycles)

            components["lut_reads"] = (lookup_ops * cost.lut_lookup_issue_cycles)

            components["base_total"] = sum(
                value
                for key, value in components.items()
                if key != "base_total"
            )

        bank_result = apply_bank_penalties(
            components,
            patterns,
            regions,
            bank_model,
            execution,
            active_cores,
            pessimism,
        )

        expected_cycles = ceil(
            bank_result[
                "cycles_expected_before_safety"
            ]
            * calibration_scale
        )

        pessimistic_cycles = ceil(
            bank_result[
                "cycles_pessimistic_before_safety"
            ]
            * pessimism.kernel_safety_factor
            * calibration_scale
        )

        core_results.append(
            {
                "core_id": item.core_id,
                "active": True,
                "implementation": ("lut" if is_lut else "mac"),
                "output_pixels": item.output_pixels,
                "macs": item.macs,
                "lut_lookups": lookup_ops,
                "output_range": {
                    "y": (
                        item.output_y_start,
                        item.output_y_stop,
                    ),
                    "x": (
                        item.output_x_start,
                        item.output_x_stop,
                    ),
                },
                "mac_lower_bound_cycles": (
                    0
                    if is_lut
                    else ceil(item.macs / peak_per_core)
                ),
                "operation_counts": counts,
                "base_components": components,
                "bank_model": bank_result,
                "expected_cycles": expected_cycles,
                "pessimistic_cycles": pessimistic_cycles,
            }
        )

    critical_expected = max(
        core_results,
        key=lambda item:
            item["expected_cycles"],
    )

    critical_pessimistic = max(
        core_results,
        key=lambda item:
            item["pessimistic_cycles"],
    )
    if is_lut:
        lower_bound = 0
    else:
        lower_bound = max(
            item["mac_lower_bound_cycles"] for item in core_results
        )

    return {
        "model": (
            "lut_operations_plus_probabilistic_banks"
            if is_lut
            else
            "source_operations_plus_probabilistic_banks"
        ),
        "implementation": ("lut" if is_lut else "mac"),
        "kernel_name": kernel.name,
        "requested_cores": execution.num_cores,
        "active_cores": active_cores,
        "peak_mac_per_cycle_per_core": peak_per_core,
        "partition_mode": partition_mode,
        "mac_lower_bound_cycles": lower_bound,
        "lut_lookup_operations": (
            sum(
                item.get("lut_lookups", 0)
                for item in core_results
            )
            if is_lut
            else 0
        ),
        "expected_cycles": critical_expected["expected_cycles"],
        "pessimistic_cycles": critical_pessimistic["pessimistic_cycles"],
        "critical_core_expected": critical_expected["core_id"],
        "critical_core_pessimistic": critical_pessimistic["core_id"],
        "core_results": core_results,
        "calibration_scale": calibration_scale,
    }