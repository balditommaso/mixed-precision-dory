from __future__ import annotations

from math import ceil
from typing import Any, Dict, List, Literal, Mapping, Sequence, Union

from .specs.factory import build_execution_spec_automatically
from .config import AutoSpecConfig, ExecutionConfig, KernelCostModel, PessimismConfig
from .descriptors import AddKernelSpec, KernelComputeSpec, LinearKernelSpec, PartitionMode, PoolKernelSpec
from .dma import estimate_dma_transfer
from .hardware import get_dma_hardware_model, get_l1_bank_model
from .kernels import (
    estimate_add_compute,
    estimate_compute,
    estimate_grouped_compute_fallback,
    estimate_linear_compute,
    estimate_pool_compute,
)


def process(
    graph: Sequence[Any], 
    hw_spec: Mapping[str, Any], 
    execution: ExecutionConfig, *, 
    auto_spec: Union[AutoSpecConfig, None]=None, 
    measured_cycles: Union[Mapping[str, int], None]=None, 
    kernel_cost: Union[KernelCostModel, None]=None, 
    pessimism: Union[PessimismConfig, None]=None, 
    peak_key: str='8bits', 
    partition_mode: PartitionMode='implementation_exact', 
    kernel_calibration_scales: Union[Mapping[str, float], None]=None, 
    global_calibration_scale: float=1.0
) -> List[Dict[str, Any]]:
    """
    Estimate every graph node without a per-node execution-spec mapping.
    """
    cost = kernel_cost or KernelCostModel()
    pess = pessimism or PessimismConfig()
    auto = auto_spec or AutoSpecConfig()
    if global_calibration_scale <= 0:
        raise ValueError('global_calibration_scale must be positive')
    dma_hw = get_dma_hardware_model(hw_spec)
    bank_model = get_l1_bank_model(hw_spec)
    cluster_peak = float(hw_spec['peak MAC/cycle'][peak_key])
    results: List[Dict[str, Any]] = []
    for node in graph:
        name = str(node.name)
        warnings: List[str] = []
        if not execution.is_power_of_two:
            warnings.append(f'NUM_CORES={execution.num_cores} is not a power of two; the generated shift/mask partition may leave cores idle')
        spec, automatic_metadata, automatic_warnings = build_execution_spec_automatically(node, auto)
        warnings.extend(automatic_warnings)
        metadata_source = str(automatic_metadata['source'])
        dma_details = [estimate_dma_transfer(transfer, execution, dma_hw) for transfer in spec.dma_transfers]
        dma_expected = sum((item['expected_cycles'] for item in dma_details))
        dma_pessimistic_raw = sum((item['pessimistic_cycles'] for item in dma_details))
        dma_pessimistic = ceil(dma_pessimistic_raw * pess.dma_safety_factor)
        calibration_key = spec.compute_kernel.name if spec.compute_kernel is not None else str(getattr(node, 'op_type', 'generic'))
        calibration_scale = global_calibration_scale * float((kernel_calibration_scales or {}).get(calibration_key, 1.0))
        if calibration_scale <= 0:
            raise ValueError(f'calibration scale for kernel {calibration_key!r} must be positive')
        if isinstance(spec.compute_kernel, KernelComputeSpec):
            modeled_macs = int(spec.compute_kernel.total_macs)
            graph_macs = int(node.MACs)
            relative_mac_error = (
                abs(modeled_macs - graph_macs) / max(1, graph_macs)
            )
            if relative_mac_error > 0.05:
                warnings.append(
                    f'kernel descriptor has {modeled_macs} MACs but graph reports '
                    f'{graph_macs}; using the graph-MAC group-aware fallback'
                )
                compute = estimate_grouped_compute_fallback(
                    spec.compute_kernel, graph_macs, hw_spec, execution, pess,
                    peak_key, calibration_scale
                )
            else:
                compute = estimate_compute(
                    spec.compute_kernel, hw_spec, execution, cost, bank_model,
                    spec.bank_access_patterns, spec.l1_regions, pess, peak_key,
                    partition_mode, calibration_scale
                )
        elif isinstance(spec.compute_kernel, PoolKernelSpec):
            compute = estimate_pool_compute(
                spec.compute_kernel, execution, cost, bank_model,
                spec.bank_access_patterns, spec.l1_regions, pess,
                calibration_scale
            )
        elif isinstance(spec.compute_kernel, LinearKernelSpec):
            graph_macs = int(node.MACs)
            if graph_macs > 0 and spec.compute_kernel.total_macs != graph_macs:
                warnings.append(
                    f'linear descriptor has {spec.compute_kernel.total_macs} MACs '
                    f'but graph reports {graph_macs}'
                )
            compute = estimate_linear_compute(
                spec.compute_kernel, hw_spec, execution, cost, bank_model,
                spec.bank_access_patterns, spec.l1_regions, pess, peak_key,
                calibration_scale
            )
        elif isinstance(spec.compute_kernel, AddKernelSpec):
            compute = estimate_add_compute(
                spec.compute_kernel, execution, cost, bank_model,
                spec.bank_access_patterns, spec.l1_regions, pess,
                calibration_scale
            )
            if compute.get('ignored_tail_values', 0):
                warnings.append(
                    'The supplied add kernel processes values in groups of four '
                    'and has no scalar tail loop.'
                )
        else:
            mac_lower = ceil(int(node.MACs) / cluster_peak)
            compute = {'model': 'mac_only_fallback', 'mac_lower_bound_cycles': mac_lower, 'expected_cycles': ceil(mac_lower * 1.25 * calibration_scale), 'pessimistic_cycles': ceil(mac_lower * 1.75 * calibration_scale), 'active_cores': execution.num_cores, 'core_results': []}
        barrier_count = spec.team_barriers_outside_kernel + spec.team_barriers_inside_kernel
        control_cycles = pess.layer_fixed_overhead_cycles + int(spec.control_events.get('kernel_calls', 0)) * pess.kernel_launch_cycles + barrier_count * pess.team_barrier_cycles + int(spec.control_events.get('dma_allocations', 0)) * pess.dma_allocate_cycles + int(spec.control_events.get('dma_frees', 0)) * pess.dma_free_cycles
        lower_bound = compute['mac_lower_bound_cycles'] + sum((ceil(item['physical_bytes'] / (dma_hw.read_bandwidth_bytes_per_cycle if item['direction'] == 'L2_TO_L1' else dma_hw.write_bandwidth_bytes_per_cycle)) for item in dma_details))
        expected_total = ceil(compute['expected_cycles'] + dma_expected + control_cycles)
        pessimistic_total = ceil(compute['pessimistic_cycles'] + dma_pessimistic + control_cycles)
        measured = None if measured_cycles is None else measured_cycles.get(name)
        diagnostics: Dict[str, Any] = {'measured_cycles': measured, 'expected_error_cycles': None, 'pessimistic_error_cycles': None, 'expected_ratio_to_measured': None, 'pessimistic_ratio_to_measured': None}
        if measured is not None and measured > 0:
            diagnostics.update({'expected_error_cycles': expected_total - measured, 'pessimistic_error_cycles': pessimistic_total - measured, 'expected_ratio_to_measured': expected_total / measured, 'pessimistic_ratio_to_measured': pessimistic_total / measured})
        for item in dma_details:
            difference = item['logical_physical_difference']
            if difference not in (None, 0):
                warnings.append(f"{item['name']} DMA transfers {item['physical_bytes']} bytes while graph metadata reports {item['logical_bytes']} bytes")
        results.append({'name': name, 'metadata_source': metadata_source, 'automatic_metadata': automatic_metadata, 'calibration_key': calibration_key, 'num_cores': execution.num_cores, 'macs': int(node.MACs), 'lower_bound_cycles': lower_bound, 'expected_cycles': expected_total, 'pessimistic_cycles': pessimistic_total, 'compute': compute, 'dma': {'transfers': dma_details, 'expected_cycles': dma_expected, 'pessimistic_cycles': dma_pessimistic}, 'control_cycles': control_cycles, 'diagnostics': diagnostics, 'warnings': warnings})
    return results


def derive_calibration_scale(result: Mapping[str, Any], measured_cycles: int, *, target: Literal['expected', 'pessimistic']='expected') -> float:
    """
    Derive a compute calibration scale from one measured layer.

    DMA and fixed control costs are held constant. Apply the returned value to
    the corresponding kernel family through ``kernel_calibration_scales``.
    """
    if measured_cycles <= 0:
        raise ValueError('measured_cycles must be positive')
    compute_key = 'expected_cycles' if target == 'expected' else 'pessimistic_cycles'
    compute_cycles = float(result['compute'][compute_key])
    non_compute = float(result['dma'][compute_key]) + float(result['control_cycles'])
    required_compute = measured_cycles - non_compute
    if compute_cycles <= 0 or required_compute <= 0:
        raise ValueError('measurement cannot produce a positive compute scale')
    return required_compute / compute_cycles
