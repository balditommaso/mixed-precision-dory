from __future__ import annotations

from math import ceil, log2
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..banks import apply_bank_penalties
from ..config import ExecutionConfig, KernelCostModel, L1BankModel, PessimismConfig
from ..descriptors import BankAccessPattern, L1RegionSpec, PoolKernelSpec


def _pool_partition_counts(work_items: int, num_cores: int) -> List[int]:
    """Mirror the max-pool kernel's row partition, including its mask quirk."""
    if work_items <= 0:
        return [0] * num_cores
    n_cores = min(num_cores, work_items)
    log2_cores = int(log2(n_cores)) if n_cores > 1 else 0
    chunk = (work_items >> log2_cores) + int(
        (work_items & (num_cores - 1)) != 0
    )
    counts: List[int] = []
    for core_id in range(num_cores):
        start = min(chunk * core_id, work_items)
        stop = min(start + chunk, work_items)
        counts.append(max(0, stop - start))
    return counts


def _pool_horizontal_values(kernel: PoolKernelSpec, rows: int) -> Tuple[int, int, int]:
    copy_values = 0
    compare_values = 0
    windows = 0
    for _ in range(rows):
        for out_x in range(kernel.output_width):
            nominal_start = out_x * kernel.stride_width - kernel.padding_left
            start = max(0, nominal_start)
            stop = min(kernel.input_width, nominal_start + kernel.kernel_width)
            width = max(0, stop - start)
            if width <= 0:
                continue
            copy_values += kernel.channels
            compare_values += max(0, width - 1) * kernel.channels
            windows += 1
    return copy_values, compare_values, windows


def _pool_vertical_values(kernel: PoolKernelSpec, output_rows: int, start_row: int) -> Tuple[int, int, int]:
    copy_values = 0
    compare_values = 0
    windows = 0
    for out_y in range(start_row, start_row + output_rows):
        nominal_start = out_y * kernel.stride_height - kernel.padding_top
        start = max(0, nominal_start)
        stop = min(kernel.input_height, nominal_start + kernel.kernel_height)
        height = max(0, stop - start)
        if height <= 0:
            continue
        row_values = kernel.output_width * kernel.channels
        copy_values += row_values
        compare_values += max(0, height - 1) * row_values
        windows += 1
    return copy_values, compare_values, windows


def _pool_components(
    copy_values: int,
    compare_values: int,
    windows: int,
    rows: int,
    cost: KernelCostModel,
) -> Dict[str, float]:
    copy_total = copy_values * cost.pool_copy_cycles_per_value
    compare_total = compare_values * cost.pool_compare_cycles_per_value
    components: Dict[str, float] = {
        # Approximate decomposition of scalar copy and vectorized compare helper.
        'input_reads': 0.50 * copy_total + 0.50 * compare_total,
        'output_writes': 0.35 * copy_total + 0.30 * compare_total,
        'arithmetic': 0.20 * compare_total,
        'control': (
            0.15 * copy_total
            + windows * cost.pool_window_setup_cycles
            + rows * cost.pool_row_setup_cycles
        ),
    }
    components['base_total'] = sum(components.values())
    return components


def _apply_pessimistic_memory_pressure(
    bank_result: Mapping[str, Any],
    component_names: Sequence[str],
    pressure_factor: float,
) -> float:
    """
    Inflate the pessimistic memory portion of a phase.

    The probabilistic bank model treats each logical stream separately.
    Pooling and quantized addition issue multiple L1 streams in the same loop
    (for example target read + window read + target write, or two input reads +
    one output write). This factor conservatively accounts for shared-port and
    instruction-issue serialization that is not captured by independent stream
    occupancy.
    """
    if pressure_factor < 1.0:
        raise ValueError('pressure_factor must be >= 1')
    components = bank_result.get('components_pessimistic', {})
    memory_cycles = sum(float(components.get(name, 0.0)) for name in component_names)
    return float(bank_result['cycles_pessimistic_before_safety']) + (pressure_factor - 1.0) * memory_cycles


def estimate_pool_compute(
    kernel: PoolKernelSpec,
    execution: ExecutionConfig,
    cost: KernelCostModel,
    bank_model: L1BankModel,
    patterns: Sequence[BankAccessPattern],
    regions: Mapping[str, L1RegionSpec],
    pessimism: PessimismConfig,
    calibration_scale: float = 1.0,
) -> Dict[str, Any]:
    horizontal_counts = _pool_partition_counts(kernel.input_height, execution.num_cores)
    vertical_counts = _pool_partition_counts(kernel.output_height, execution.num_cores)
    active_horizontal = sum(count > 0 for count in horizontal_counts)
    active_vertical = sum(count > 0 for count in vertical_counts)

    horizontal_results: List[Dict[str, Any]] = []
    for core_id, rows in enumerate(horizontal_counts):
        if rows <= 0:
            horizontal_results.append({'core_id': core_id, 'rows': 0, 'expected_cycles': 0, 'pessimistic_cycles': 0})
            continue
        copy_values, compare_values, windows = _pool_horizontal_values(kernel, rows)
        components = _pool_components(copy_values, compare_values, windows, rows, cost)
        bank = apply_bank_penalties(
            components, patterns, regions, bank_model, execution,
            max(1, active_horizontal), pessimism,
        )
        horizontal_results.append({
            'core_id': core_id,
            'rows': rows,
            'copy_values': copy_values,
            'compare_values': compare_values,
            'base_components': components,
            'bank_model': bank,
            'expected_cycles': ceil(bank['cycles_expected_before_safety'] * calibration_scale),
            'pessimistic_cycles': ceil(
                _apply_pessimistic_memory_pressure(
                    bank,
                    ('input_reads', 'output_writes'),
                    pessimism.pooling_memory_pressure_factor,
                )
                * pessimism.pooling_kernel_safety_factor
                * calibration_scale
            ),
        })

    vertical_results: List[Dict[str, Any]] = []
    cursor = 0
    for core_id, rows in enumerate(vertical_counts):
        if rows <= 0:
            vertical_results.append({'core_id': core_id, 'rows': 0, 'expected_cycles': 0, 'pessimistic_cycles': 0})
            continue
        copy_values, compare_values, windows = _pool_vertical_values(kernel, rows, cursor)
        components = _pool_components(copy_values, compare_values, windows, rows, cost)
        bank = apply_bank_penalties(
            components, patterns, regions, bank_model, execution,
            max(1, active_vertical), pessimism,
        )
        vertical_results.append({
            'core_id': core_id,
            'rows': rows,
            'row_range': (cursor, cursor + rows),
            'copy_values': copy_values,
            'compare_values': compare_values,
            'base_components': components,
            'bank_model': bank,
            'expected_cycles': ceil(bank['cycles_expected_before_safety'] * calibration_scale),
            'pessimistic_cycles': ceil(
                _apply_pessimistic_memory_pressure(
                    bank,
                    ('input_reads', 'output_writes'),
                    pessimism.pooling_memory_pressure_factor,
                )
                * pessimism.pooling_kernel_safety_factor
                * calibration_scale
            ),
        })
        cursor += rows

    h_expected = max(item['expected_cycles'] for item in horizontal_results)
    h_pessimistic = max(item['pessimistic_cycles'] for item in horizontal_results)
    v_expected = max(item['expected_cycles'] for item in vertical_results)
    v_pessimistic = max(item['pessimistic_cycles'] for item in vertical_results)

    total_copy = sum(item.get('copy_values', 0) for item in horizontal_results + vertical_results)
    total_compare = sum(item.get('compare_values', 0) for item in horizontal_results + vertical_results)
    active_max = max(1, active_horizontal, active_vertical)
    operation_lower_bound = ceil((total_copy + total_compare) / (4 * active_max))

    return {
        'model': 'two_pass_maxpool_source_operations',
        'kernel_name': kernel.name,
        'requested_cores': execution.num_cores,
        'active_cores': max(active_horizontal, active_vertical),
        'active_cores_horizontal': active_horizontal,
        'active_cores_vertical': active_vertical,
        'mac_lower_bound_cycles': operation_lower_bound,
        'expected_cycles': h_expected + v_expected,
        'pessimistic_cycles': h_pessimistic + v_pessimistic,
        'horizontal_results': horizontal_results,
        'vertical_results': vertical_results,
        'calibration_scale': calibration_scale,
    }
