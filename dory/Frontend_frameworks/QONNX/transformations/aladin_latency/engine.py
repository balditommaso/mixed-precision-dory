from __future__ import annotations

from math import ceil
from typing import Any, Dict, List, Literal, Mapping, Sequence, Union

from .config import (
    AutoSpecConfig,
    ExecutionConfig,
    KernelCostModel,
    PessimismConfig,
    TilingModelConfig,
)
from .descriptors import (
    AddKernelSpec,
    ComputeKernelSpec,
    DMATransferSpec,
    KernelComputeSpec,
    LinearKernelSpec,
    NodeExecutionSpec,
    PartitionMode,
    PoolKernelSpec,
)
from .dma import estimate_dma_transfer
from .hardware import get_dma_hardware_model, get_l1_bank_model
from .kernels import (
    estimate_add_compute,
    estimate_compute,
    estimate_grouped_compute_fallback,
    estimate_linear_compute,
    estimate_pool_compute,
)
from .specs.factory import build_execution_spec_automatically
from .tiling import (
    TilePlan,
    build_tile_plan,
    level_is_tiled,
    level_memory_footprint,
    tile_regions,
    total_role_bytes,
    transfer_repetitions,
)



def _estimate_kernel(
    kernel: Union[ComputeKernelSpec, None],
    *,
    node: Any,
    hw_spec: Mapping[str, Any],
    execution: ExecutionConfig,
    cost: KernelCostModel,
    pessimism: PessimismConfig,
    bank_model,
    patterns,
    regions,
    peak_key: str,
    partition_mode: PartitionMode,
    calibration_scale: float,
    validate_graph_macs: bool,
    warnings: List[str],
) -> Dict[str, Any]:
    cluster_peak = float(hw_spec['peak MAC/cycle'][peak_key])

    if isinstance(kernel, KernelComputeSpec):
        if validate_graph_macs:
            modeled_macs = int(kernel.total_macs)
            graph_macs = int(getattr(node, 'MACs', 0) or 0)
            relative_mac_error = abs(modeled_macs - graph_macs) / max(1, graph_macs)
            if graph_macs > 0 and relative_mac_error > 0.05:
                warnings.append(
                    f'kernel descriptor has {modeled_macs} MACs but graph reports '
                    f'{graph_macs}; using the graph-MAC group-aware fallback'
                )
                return estimate_grouped_compute_fallback(
                    kernel,
                    graph_macs,
                    hw_spec,
                    execution,
                    pessimism,
                    peak_key,
                    calibration_scale,
                )
        return estimate_compute(
            kernel,
            hw_spec,
            execution,
            cost,
            bank_model,
            patterns,
            regions,
            pessimism,
            peak_key,
            partition_mode,
            calibration_scale,
        )

    if isinstance(kernel, PoolKernelSpec):
        return estimate_pool_compute(
            kernel,
            execution,
            cost,
            bank_model,
            patterns,
            regions,
            pessimism,
            calibration_scale,
        )

    if isinstance(kernel, LinearKernelSpec):
        if validate_graph_macs:
            graph_macs = int(getattr(node, 'MACs', 0) or 0)
            if graph_macs > 0 and kernel.total_macs != graph_macs:
                warnings.append(
                    f'linear descriptor has {kernel.total_macs} MACs '
                    f'but graph reports {graph_macs}'
                )
        return estimate_linear_compute(
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

    if isinstance(kernel, AddKernelSpec):
        compute = estimate_add_compute(
            kernel,
            execution,
            cost,
            bank_model,
            patterns,
            regions,
            pessimism,
            calibration_scale,
        )
        if compute.get('ignored_tail_values', 0):
            warnings.append(
                'The supplied add kernel processes values in groups of four '
                'and has no scalar tail loop.'
            )
        return compute

    mac_lower = ceil(int(getattr(node, 'MACs', 0) or 0) / cluster_peak)
    return {
        'model': 'mac_only_fallback',
        'mac_lower_bound_cycles': mac_lower,
        'expected_cycles': ceil(mac_lower * 1.25 * calibration_scale),
        'pessimistic_cycles': ceil(mac_lower * 1.75 * calibration_scale),
        'active_cores': execution.num_cores,
        'core_results': [],
    }



def _work_units(kernel: ComputeKernelSpec) -> int:
    if isinstance(kernel, KernelComputeSpec):
        return kernel.total_macs
    if isinstance(kernel, LinearKernelSpec):
        return kernel.total_macs
    if isinstance(kernel, PoolKernelSpec):
        return kernel.output_values
    if isinstance(kernel, AddKernelSpec):
        return kernel.total_values
    return 0



def _estimate_tiled_compute(
    *,
    plan: TilePlan,
    full_compute: Dict[str, Any],
    spec: NodeExecutionSpec,
    node: Any,
    hw_spec: Mapping[str, Any],
    execution: ExecutionConfig,
    cost: KernelCostModel,
    pessimism: PessimismConfig,
    tiling: TilingModelConfig,
    bank_model,
    peak_key: str,
    partition_mode: PartitionMode,
    calibration_scale: float,
    warnings: List[str],
) -> Dict[str, Any]:
    if not plan.enabled:
        return full_compute

    # Robust default. Tiling does not change the number of useful operations in
    # the layer. It changes DMA traffic, launch/barrier overhead, and sometimes
    # core utilization. Reconstructing the latter from generic DORY metadata is
    # target-dependent, so preserve full-layer compute unless explicitly asked
    # for shape-aware tile compute.
    if tiling.compute_mode == 'full_layer':
        result = dict(full_compute)
        result['model'] = 'full_layer_compute_plus_tiled_overhead_' + str(
            full_compute.get('model', 'kernel')
        )
        result['tiling_mode'] = 'full_layer'
        result['tile_groups'] = []
        result['derived_total_tiles'] = plan.derived_total_tiles
        result['runtime_total_tiles'] = plan.runtime_total_tiles
        return result

    if not plan.groups:
        result = dict(full_compute)
        result['model'] = 'tiled_count_only_' + str(full_compute.get('model', 'kernel'))
        result['pessimistic_cycles'] = ceil(
            float(full_compute['pessimistic_cycles'])
            * tiling.unknown_geometry_compute_factor
        )
        result['tiling_mode'] = 'count_only'
        result['tile_groups'] = []
        return result

    expected = 0.0
    pessimistic = 0.0
    lower = 0.0
    tiled_units = 0
    details: List[Dict[str, Any]] = []

    for group_index, group in enumerate(plan.groups):
        regions = tile_regions(spec.l1_regions, group)
        tile_compute = _estimate_kernel(
            group.kernel,
            node=node,
            hw_spec=hw_spec,
            execution=execution,
            cost=cost,
            pessimism=pessimism,
            bank_model=bank_model,
            patterns=spec.bank_access_patterns,
            regions=regions,
            peak_key=peak_key,
            partition_mode=partition_mode,
            calibration_scale=calibration_scale,
            validate_graph_macs=False,
            warnings=warnings,
        )
        multiplicity = int(group.multiplicity)
        expected += float(tile_compute['expected_cycles']) * multiplicity
        pessimistic += float(tile_compute['pessimistic_cycles']) * multiplicity
        lower += float(tile_compute['mac_lower_bound_cycles']) * multiplicity
        tiled_units += _work_units(group.kernel) * multiplicity
        details.append(
            {
                'group_index': group_index,
                'multiplicity': multiplicity,
                'shape': dict(group.shape),
                'role_bytes': dict(group.role_bytes),
                'expected_cycles_per_tile': int(tile_compute['expected_cycles']),
                'pessimistic_cycles_per_tile': int(tile_compute['pessimistic_cycles']),
                'active_cores': int(tile_compute.get('active_cores', execution.num_cores)),
            }
        )

    full_units = _work_units(spec.compute_kernel) if spec.compute_kernel is not None else 0
    # For convolution/linear, force the tile sum to represent exactly the graph
    # arithmetic workload. This protects against incomplete channel metadata.
    if isinstance(spec.compute_kernel, (KernelComputeSpec, LinearKernelSpec)):
        graph_macs = int(getattr(node, 'MACs', 0) or 0)
        target_units = graph_macs if graph_macs > 0 else full_units
        if tiled_units > 0 and target_units > 0:
            scale = target_units / float(tiled_units)
            if abs(scale - 1.0) > 0.05:
                warnings.append(
                    'Tile descriptors cover %d MACs while the graph reports %d; '
                    'tile compute is rescaled by %.3f.'
                    % (tiled_units, target_units, scale)
                )
            expected *= scale
            pessimistic *= scale
            lower *= scale

    pessimistic *= tiling.edge_tile_safety_factor

    return {
        'model': 'shape_aware_tiled_' + str(full_compute.get('model', 'kernel')),
        'mac_lower_bound_cycles': ceil(lower),
        'expected_cycles': ceil(expected),
        'pessimistic_cycles': ceil(pessimistic),
        'active_cores': max((item['active_cores'] for item in details), default=0),
        'core_results': [],
        'tiling_mode': 'shape_aware',
        'tile_groups': details,
        'derived_total_tiles': plan.derived_total_tiles,
        'runtime_total_tiles': plan.runtime_total_tiles,
    }



def _tile_aware_transfers(
    spec: NodeExecutionSpec,
    plan: TilePlan,
    tiling: TilingModelConfig,
) -> Sequence[DMATransferSpec]:
    if not plan.enabled:
        return spec.dma_transfers

    role_totals = total_role_bytes(plan)
    result: List[DMATransferSpec] = []

    for transfer in spec.dma_transfers:
        base_bytes = max(0, int(transfer.physical_bytes))
        repetitions = transfer_repetitions(plan, transfer.name)

        if plan.groups:
            reconstructed = int(role_totals.get(transfer.name, base_bytes))
            total_bytes = max(base_bytes, reconstructed)

            # Payload reconstruction is still approximate because exact loop
            # ordering and residency are not encoded in HW_node.  Bound only
            # redundant payload; role-specific command counts remain exact with
            # respect to the dimensional loop model.
            if base_bytes > 0:
                total_bytes = min(
                    total_bytes,
                    ceil(base_bytes * tiling.max_dma_redundancy_factor),
                )

            # Useful output data is written once per output tile, not once per
            # input-channel reduction tile.  The role-specific repetition count
            # changes command/startup cost while preserving total output bytes.
            if transfer.name == 'output':
                total_bytes = base_bytes
        else:
            total_bytes = ceil(
                base_bytes * tiling.unknown_geometry_dma_redundancy_factor
            )
            repetitions = max(1, plan.runtime_total_tiles)

        per_submission = max(1, ceil(total_bytes / float(max(1, repetitions))))
        result.append(
            DMATransferSpec(
                name=transfer.name,
                direction=transfer.direction,
                number_of_2d_copies=1,
                number_of_1d_copies=1,
                length_1d_copy=per_submission,
                stride_2d=per_submission,
                stride_1d=per_submission,
                hwc_to_chw=transfer.hwc_to_chw,
                logical_bytes=transfer.logical_bytes,
                submissions=max(1, repetitions),
                barrier_calls=transfer.barrier_calls * max(1, repetitions),
                physical_bytes_override=total_bytes,
            )
        )
    return tuple(result)


def _control_cycles(
    spec: NodeExecutionSpec,
    plan: TilePlan,
    pessimism: PessimismConfig,
) -> Dict[str, int]:
    if spec.compute_kernel is None:
        kernel_calls = 0
    elif plan.enabled:
        kernel_calls = max(1, int(plan.runtime_total_tiles))
    else:
        kernel_calls = 1

    # Source parsing yields call sites, not dynamic executions.  The dynamic
    # kernel count is derived from the dimensional tile loops instead.
    kernel_call_sites = int(spec.control_events.get('kernel_calls', 0))

    # Internal barriers execute once per kernel invocation.  External barriers
    # are kept static unless the generated source parser can prove that they are
    # located inside a runtime tile loop.
    outside_barriers = int(spec.team_barriers_outside_kernel)
    inside_barriers = int(spec.team_barriers_inside_kernel) * kernel_calls
    allocations = int(spec.control_events.get('dma_allocations', 0))
    frees = int(spec.control_events.get('dma_frees', 0))

    total = (
        pessimism.layer_fixed_overhead_cycles
        + kernel_calls * pessimism.kernel_launch_cycles
        + (outside_barriers + inside_barriers) * pessimism.team_barrier_cycles
        + allocations * pessimism.dma_allocate_cycles
        + frees * pessimism.dma_free_cycles
    )
    return {
        'cycles': int(total),
        'runtime_tiles': max(1, int(plan.runtime_total_tiles)) if plan.enabled else 1,
        'kernel_calls': kernel_calls,
        'kernel_call_sites': kernel_call_sites,
        'outside_barriers': outside_barriers,
        'inside_barriers': inside_barriers,
        'dma_allocations': allocations,
        'dma_frees': frees,
    }


def process(
    graph: Sequence[Any],
    hw_spec: Mapping[str, Any],
    execution: ExecutionConfig,
    *,
    auto_spec: Union[AutoSpecConfig, None] = None,
    measured_cycles: Union[Mapping[str, int], None] = None,
    kernel_cost: Union[KernelCostModel, None] = None,
    pessimism: Union[PessimismConfig, None] = None,
    tiling: Union[TilingModelConfig, None] = None,
    peak_key: str = '8bits',
    partition_mode: PartitionMode = 'implementation_exact',
    kernel_calibration_scales: Union[Mapping[str, float], None] = None,
    global_calibration_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    """Estimate every graph node, including DORY L1 tile execution."""

    cost = kernel_cost or KernelCostModel()
    pess = pessimism or PessimismConfig()
    tile_config = tiling or TilingModelConfig()
    auto = auto_spec or AutoSpecConfig()
    if global_calibration_scale <= 0:
        raise ValueError('global_calibration_scale must be positive')

    dma_hw = get_dma_hardware_model(hw_spec)
    bank_model = get_l1_bank_model(hw_spec)
    results: List[Dict[str, Any]] = []

    for node in graph:
        name = str(node.name)
        warnings: List[str] = []
        if not execution.is_power_of_two:
            warnings.append(
                f'NUM_CORES={execution.num_cores} is not a power of two; '
                'the generated shift/mask partition may leave cores idle'
            )

        spec, automatic_metadata, automatic_warnings = (
            build_execution_spec_automatically(node, auto)
        )
        warnings.extend(automatic_warnings)
        metadata_source = str(automatic_metadata['source'])

        plan = build_tile_plan(node, spec, automatic_metadata, tile_config)
        warnings.extend(plan.warnings)

        l1_capacity = int(hw_spec.get('memory', {}).get('L1', {}).get('dimension', 0) or 0)
        l2_capacity = int(hw_spec.get('memory', {}).get('L2', {}).get('dimension', 0) or 0)
        l1_footprint = level_memory_footprint(node, 'L1')
        l2_footprint = level_memory_footprint(node, 'L2')
        if l1_capacity > 0 and l1_footprint > l1_capacity:
            warnings.append(
                f'L1 tile footprint {l1_footprint} exceeds configured L1 capacity '
                f'{l1_capacity}; graph/code may not match the hardware configuration.'
            )
        if l2_capacity > 0 and l2_footprint > l2_capacity:
            warnings.append(
                f'L2 tile footprint {l2_footprint} exceeds configured L2 capacity '
                f'{l2_capacity}; graph/code may not match the hardware configuration.'
            )
        if level_is_tiled(node, spec.compute_kernel, 'L2') and not (
            hw_spec.get('latency_model', {}).get('dma_l3_l2')
        ):
            warnings.append(
                'The node is tiled at L2, but no latency_model.dma_l3_l2 '
                'configuration is present. L3--L2 traffic is not included and '
                'the estimate can remain optimistic.'
            )

        calibration_key = (
            spec.compute_kernel.name
            if spec.compute_kernel is not None
            else str(getattr(node, 'op_type', 'generic'))
        )
        calibration_scale = global_calibration_scale * float(
            (kernel_calibration_scales or {}).get(calibration_key, 1.0)
        )
        if calibration_scale <= 0:
            raise ValueError(
                f'calibration scale for kernel {calibration_key!r} must be positive'
            )

        full_compute = _estimate_kernel(
            spec.compute_kernel,
            node=node,
            hw_spec=hw_spec,
            execution=execution,
            cost=cost,
            pessimism=pess,
            bank_model=bank_model,
            patterns=spec.bank_access_patterns,
            regions=spec.l1_regions,
            peak_key=peak_key,
            partition_mode=partition_mode,
            calibration_scale=calibration_scale,
            validate_graph_macs=True,
            warnings=warnings,
        )
        compute = _estimate_tiled_compute(
            plan=plan,
            full_compute=full_compute,
            spec=spec,
            node=node,
            hw_spec=hw_spec,
            execution=execution,
            cost=cost,
            pessimism=pess,
            tiling=tile_config,
            bank_model=bank_model,
            peak_key=peak_key,
            partition_mode=partition_mode,
            calibration_scale=calibration_scale,
            warnings=warnings,
        )

        transfers = _tile_aware_transfers(spec, plan, tile_config)
        dma_details = [
            estimate_dma_transfer(transfer, execution, dma_hw)
            for transfer in transfers
        ]
        dma_expected = sum(item['expected_cycles'] for item in dma_details)
        dma_pessimistic_raw = sum(
            item['pessimistic_cycles'] for item in dma_details
        )
        dma_pessimistic = ceil(dma_pessimistic_raw * pess.dma_safety_factor)

        control = _control_cycles(spec, plan, pess)
        control_cycles = control['cycles']

        lower_bound = compute['mac_lower_bound_cycles'] + sum(
            ceil(
                item['physical_bytes']
                / (
                    dma_hw.read_bandwidth_bytes_per_cycle
                    if item['direction'] == 'L2_TO_L1'
                    else dma_hw.write_bandwidth_bytes_per_cycle
                )
            )
            for item in dma_details
        )
        expected_total = ceil(
            compute['expected_cycles'] + dma_expected + control_cycles
        )
        pessimistic_total = ceil(
            compute['pessimistic_cycles'] + dma_pessimistic + control_cycles
        )

        measured = None if measured_cycles is None else measured_cycles.get(name)
        diagnostics: Dict[str, Any] = {
            'measured_cycles': measured,
            'expected_error_cycles': None,
            'pessimistic_error_cycles': None,
            'expected_ratio_to_measured': None,
            'pessimistic_ratio_to_measured': None,
        }
        if measured is not None and measured > 0:
            diagnostics.update(
                {
                    'expected_error_cycles': expected_total - measured,
                    'pessimistic_error_cycles': pessimistic_total - measured,
                    'expected_ratio_to_measured': expected_total / measured,
                    'pessimistic_ratio_to_measured': pessimistic_total / measured,
                }
            )

        for item in dma_details:
            difference = item['logical_physical_difference']
            if difference not in (None, 0):
                warnings.append(
                    f"{item['name']} DMA transfers {item['physical_bytes']} bytes "
                    f"while graph metadata reports {item['logical_bytes']} bytes"
                )

        tiling_diagnostics = {
            'enabled': plan.enabled,
            'source': plan.source,
            'level_name': plan.level_name,
            'parent_level_name': plan.parent_level_name,
            'parsed_total_tiles': plan.parsed_total_tiles,
            'derived_total_tiles': plan.derived_total_tiles,
            'runtime_total_tiles': plan.runtime_total_tiles,
            'loop_counts': {
                'output_height': plan.loop_counts.output_height,
                'output_width': plan.loop_counts.output_width,
                'output_channels': plan.loop_counts.output_channels,
                'input_channels': plan.loop_counts.input_channels,
                'channels_coupled': plan.loop_counts.channels_coupled,
                'spatial_tiles': plan.loop_counts.spatial_tiles,
                'kernel_calls': plan.loop_counts.kernel_calls,
            },
            'inner_loop_counts': {
                'output_height': plan.inner_loop_counts.output_height,
                'output_width': plan.inner_loop_counts.output_width,
                'output_channels': plan.inner_loop_counts.output_channels,
                'input_channels': plan.inner_loop_counts.input_channels,
                'channels_coupled': plan.inner_loop_counts.channels_coupled,
            },
            'outer_loop_counts': {
                'output_height': plan.outer_loop_counts.output_height,
                'output_width': plan.outer_loop_counts.output_width,
                'output_channels': plan.outer_loop_counts.output_channels,
                'input_channels': plan.outer_loop_counts.input_channels,
                'channels_coupled': plan.outer_loop_counts.channels_coupled,
            },
            'transfer_repetitions': {
                role: transfer_repetitions(plan, role)
                for role in ('input', 'second_input', 'weights', 'bias', 'constants', 'output')
            },
            'shape_aware': plan.has_explicit_shapes,
            'role_physical_bytes': total_role_bytes(plan),
            'control': control,
        }
        automatic_metadata = dict(automatic_metadata)
        automatic_metadata['effective_total_tiles'] = plan.runtime_total_tiles

        results.append(
            {
                'name': name,
                'metadata_source': metadata_source,
                'automatic_metadata': automatic_metadata,
                'calibration_key': calibration_key,
                'num_cores': execution.num_cores,
                'macs': int(getattr(node, 'MACs', 0) or 0),
                'lower_bound_cycles': lower_bound,
                'expected_cycles': expected_total,
                'pessimistic_cycles': pessimistic_total,
                'compute': compute,
                'dma': {
                    'transfers': dma_details,
                    'expected_cycles': dma_expected,
                    'pessimistic_cycles': dma_pessimistic,
                },
                'control_cycles': control_cycles,
                'tiling': tiling_diagnostics,
                'diagnostics': diagnostics,
                'warnings': warnings,
            }
        )

    return results



def derive_calibration_scale(
    result: Mapping[str, Any],
    measured_cycles: int,
    *,
    target: Literal['expected', 'pessimistic'] = 'expected',
) -> float:
    """Derive a compute-only calibration scale from one measured layer."""

    if measured_cycles <= 0:
        raise ValueError('measured_cycles must be positive')
    compute_key = 'expected_cycles' if target == 'expected' else 'pessimistic_cycles'
    compute_cycles = float(result['compute'][compute_key])
    non_compute = float(result['dma'][compute_key]) + float(result['control_cycles'])
    required_compute = measured_cycles - non_compute
    if compute_cycles <= 0 or required_compute <= 0:
        raise ValueError('measurement cannot produce a positive compute scale')
    return required_compute / compute_cycles
