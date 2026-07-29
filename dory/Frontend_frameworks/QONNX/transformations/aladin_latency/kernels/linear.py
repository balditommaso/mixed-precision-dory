from __future__ import annotations

from math import ceil, log2
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..banks import apply_bank_penalties
from ..config import ExecutionConfig, KernelCostModel, L1BankModel, PessimismConfig
from ..descriptors import BankAccessPattern, L1RegionSpec, LinearKernelSpec


def _linear_partition_ranges(output_neurons: int, num_cores: int) -> List[Tuple[int, int]]:
    if output_neurons <= 0:
        return [(0, 0)] * num_cores
    log2_cores = int(log2(num_cores)) if num_cores > 1 else 0
    chunk = (output_neurons >> log2_cores) + int(
        (output_neurons & (num_cores - 1)) != 0
    )
    ranges: List[Tuple[int, int]] = []
    for core_id in range(num_cores):
        start = min(chunk * core_id, output_neurons)
        stop = min(start + chunk, output_neurons)
        ranges.append((start, stop))
    return ranges


def _linear_components_for_core(
    kernel: LinearKernelSpec,
    neurons: int,
    cost: KernelCostModel,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    pairs = neurons // 2
    odd = neurons % 2
    vector_iterations = kernel.input_features // 4
    scalar_tail = kernel.input_features % 4

    input_vector_loads = pairs * vector_iterations + odd * vector_iterations
    weight_vector_loads = pairs * vector_iterations * 2 + odd * vector_iterations
    dotp4 = pairs * vector_iterations * 2 + odd * vector_iterations
    input_scalar_loads = pairs * scalar_tail + odd * scalar_tail
    weight_scalar_loads = pairs * scalar_tail * 2 + odd * scalar_tail
    scalar_macs = pairs * scalar_tail * 2 + odd * scalar_tail
    outputs = neurons
    bias_loads = outputs if kernel.has_bias else 0

    counts = {
        'neurons': neurons,
        'pairs': pairs,
        'odd_neurons': odd,
        'vector_iterations': vector_iterations,
        'scalar_tail': scalar_tail,
        'input_vector_loads': input_vector_loads,
        'weight_vector_loads': weight_vector_loads,
        'dotp4': dotp4,
        'input_scalar_loads': input_scalar_loads,
        'weight_scalar_loads': weight_scalar_loads,
        'scalar_macs': scalar_macs,
        'bias_loads': bias_loads,
        'outputs': outputs,
    }
    components: Dict[str, float] = {
        'arithmetic': dotp4 * cost.dotp4_cycles + scalar_macs * cost.scalar_mac_cycles,
        'input_reads': (
            input_vector_loads * cost.linear_input_vector_load_cycles
            + input_scalar_loads * cost.linear_input_scalar_load_cycles
        ),
        'weight_reads': (
            weight_vector_loads * cost.linear_weight_vector_load_cycles
            + weight_scalar_loads * cost.linear_weight_scalar_load_cycles
        ),
        'bias_reads': bias_loads * cost.bias_load_cycles,
        'quantization': outputs * cost.quant_relu_cycles_per_output,
        'output_writes': outputs * cost.output_store_cycles,
        'control': (
            pairs * cost.linear_neuron_pair_setup_cycles
            + odd * cost.linear_single_neuron_setup_cycles
            + (pairs + odd) * vector_iterations * cost.linear_pair_loop_overhead_cycles
            + (pairs + odd) * scalar_tail * cost.linear_tail_loop_overhead_cycles
        ),
    }
    components['base_total'] = sum(components.values())
    return counts, components


def estimate_linear_compute(
    kernel: LinearKernelSpec,
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
    ranges = _linear_partition_ranges(kernel.output_neurons, execution.num_cores)
    active_cores = sum(stop > start for start, stop in ranges)
    cluster_peak = float(hw_spec['peak MAC/cycle'][peak_key])
    reference_cores = int(hw_spec.get('compute_model', {}).get('reference_cores', 8))
    peak_per_core = cluster_peak / max(1, reference_cores)

    core_results: List[Dict[str, Any]] = []
    for core_id, (start, stop) in enumerate(ranges):
        neurons = max(0, stop - start)
        if neurons <= 0:
            core_results.append({
                'core_id': core_id, 'active': False, 'neuron_range': (start, stop),
                'macs': 0, 'mac_lower_bound_cycles': 0,
                'expected_cycles': 0, 'pessimistic_cycles': 0,
            })
            continue
        counts, components = _linear_components_for_core(kernel, neurons, cost)
        bank = apply_bank_penalties(
            components, patterns, regions, bank_model, execution,
            max(1, active_cores), pessimism,
        )
        macs = neurons * kernel.input_features
        core_results.append({
            'core_id': core_id,
            'active': True,
            'neuron_range': (start, stop),
            'neurons': neurons,
            'macs': macs,
            'mac_lower_bound_cycles': ceil(macs / max(1e-9, peak_per_core)),
            'operation_counts': counts,
            'base_components': components,
            'bank_model': bank,
            'expected_cycles': ceil(bank['cycles_expected_before_safety'] * calibration_scale),
            'pessimistic_cycles': ceil(
                bank['cycles_pessimistic_before_safety']
                * pessimism.linear_kernel_safety_factor
                * calibration_scale
            ),
        })

    critical_expected = max(core_results, key=lambda item: item['expected_cycles'])
    critical_pessimistic = max(core_results, key=lambda item: item['pessimistic_cycles'])
    return {
        'model': 'linear_source_operations_plus_probabilistic_banks',
        'kernel_name': kernel.name,
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
