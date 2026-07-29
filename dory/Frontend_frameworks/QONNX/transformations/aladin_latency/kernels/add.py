from __future__ import annotations

from math import ceil
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..banks import apply_bank_penalties
from ..config import ExecutionConfig, KernelCostModel, L1BankModel, PessimismConfig
from ..descriptors import AddKernelSpec, BankAccessPattern, L1RegionSpec
from .pooling import _apply_pessimistic_memory_pressure, _pool_partition_counts


def _add_components_for_values(
    kernel: AddKernelSpec,
    values: int,
    cost: KernelCostModel,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    # The supplied implementation processes exactly four values per loop and
    # does not contain a scalar cleanup loop.
    groups4 = values >> 2
    processed_values = groups4 << 2
    ignored_tail = values - processed_values

    input_requant = (
        2.0 * processed_values * cost.add_input_requant_cycles_per_operand
    )
    input_clips = 2.0 * processed_values * cost.add_clip_cycles_per_value
    sums = processed_values * cost.add_sum_cycles_per_value
    output_requant = (
        processed_values * cost.add_output_requant_cycles_per_value
        if kernel.out_requant else 0.0
    )
    output_clips = processed_values * cost.add_clip_cycles_per_value

    counts = {
        'values': values,
        'groups4': groups4,
        'processed_values': processed_values,
        'ignored_tail': ignored_tail,
        'out_requant': int(kernel.out_requant),
    }
    components: Dict[str, float] = {
        'input_reads': processed_values * cost.add_input_load_cycles_per_value,
        'second_input_reads': processed_values * cost.add_input_load_cycles_per_value,
        'arithmetic': input_requant + input_clips + sums + output_requant + output_clips,
        'output_writes': processed_values * cost.add_output_store_cycles_per_value,
        'control': groups4 * cost.add_group4_overhead_cycles,
    }
    components['base_total'] = sum(components.values())
    return counts, components


def estimate_add_compute(
    kernel: AddKernelSpec,
    execution: ExecutionConfig,
    cost: KernelCostModel,
    bank_model: L1BankModel,
    patterns: Sequence[BankAccessPattern],
    regions: Mapping[str, L1RegionSpec],
    pessimism: PessimismConfig,
    calibration_scale: float = 1.0,
) -> Dict[str, Any]:
    # The add kernel uses the same height partition and NUM_CORES-mask quirk
    # as the supplied max-pooling implementation.
    row_counts = _pool_partition_counts(kernel.height, execution.num_cores)
    active_cores = sum(rows > 0 for rows in row_counts)

    core_results: List[Dict[str, Any]] = []
    for core_id, rows in enumerate(row_counts):
        values = rows * kernel.width * kernel.channels
        if values <= 0:
            core_results.append({
                'core_id': core_id,
                'active': False,
                'rows': 0,
                'values': 0,
                'operation_lower_bound_cycles': 0,
                'expected_cycles': 0,
                'pessimistic_cycles': 0,
            })
            continue

        counts, components = _add_components_for_values(kernel, values, cost)
        bank = apply_bank_penalties(
            components, patterns, regions, bank_model, execution,
            max(1, active_cores), pessimism,
        )
        core_results.append({
            'core_id': core_id,
            'active': True,
            'rows': rows,
            'values': values,
            'operation_counts': counts,
            'base_components': components,
            'bank_model': bank,
            'operation_lower_bound_cycles': ceil(counts['processed_values'] / 4),
            'expected_cycles': ceil(
                bank['cycles_expected_before_safety'] * calibration_scale
            ),
            'pessimistic_cycles': ceil(
                _apply_pessimistic_memory_pressure(
                    bank,
                    ('input_reads', 'second_input_reads', 'output_writes'),
                    pessimism.add_memory_pressure_factor,
                )
                * pessimism.add_kernel_safety_factor
                * calibration_scale
            ),
        })

    critical_expected = max(core_results, key=lambda item: item['expected_cycles'])
    critical_pessimistic = max(core_results, key=lambda item: item['pessimistic_cycles'])
    ignored_tail = sum(
        item.get('operation_counts', {}).get('ignored_tail', 0)
        for item in core_results
    )
    return {
        'model': 'quantized_add_source_operations_plus_probabilistic_banks',
        'kernel_name': kernel.name,
        'requested_cores': execution.num_cores,
        'active_cores': active_cores,
        'out_requant': kernel.out_requant,
        'ignored_tail_values': ignored_tail,
        'mac_lower_bound_cycles': max(
            item['operation_lower_bound_cycles'] for item in core_results
        ),
        'expected_cycles': critical_expected['expected_cycles'],
        'pessimistic_cycles': critical_pessimistic['pessimistic_cycles'],
        'critical_core_expected': critical_expected['core_id'],
        'critical_core_pessimistic': critical_pessimistic['core_id'],
        'core_results': core_results,
        'calibration_scale': calibration_scale,
    }
