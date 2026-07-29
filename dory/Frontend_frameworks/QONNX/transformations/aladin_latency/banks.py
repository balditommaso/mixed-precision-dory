from __future__ import annotations

from math import ceil, floor
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from .config import ExecutionConfig, L1BankModel, PessimismConfig
from .descriptors import BankAccessPattern, L1RegionSpec


def l1_bank_index(address: int, bank_model: L1BankModel) -> int:
    return address // bank_model.interleaving_bytes % bank_model.bank_count


def l1_banks_touched(address: int, size_bytes: int, bank_model: L1BankModel) -> Tuple[int, ...]:
    if size_bytes <= 0:
        return ()
    first = address // bank_model.interleaving_bytes
    last = (address + size_bytes - 1) // bank_model.interleaving_bytes
    return tuple((index % bank_model.bank_count for index in range(first, last + 1)))


def _resolve_requesters(pattern: BankAccessPattern, execution: ExecutionConfig, active_cores: int) -> int:
    if pattern.requester_scope == 'active_cores':
        requesters = active_cores
    elif pattern.requester_scope == 'all_cores':
        requesters = execution.num_cores
    elif pattern.requester_scope == 'single':
        requesters = 1
    elif pattern.requester_scope == 'fixed':
        if pattern.fixed_requesters is None:
            raise ValueError(f'{pattern.name}: fixed_requesters is required')
        requesters = pattern.fixed_requesters
    else:
        raise ValueError(f'unsupported requester scope: {pattern.requester_scope}')
    return requesters + pattern.concurrent_dma_requesters


def _banks_for_pattern(pattern: BankAccessPattern, regions: Mapping[str, L1RegionSpec], bank_model: L1BankModel, active_cores: int) -> Tuple[int, ...]:
    if pattern.effective_banks_override is not None:
        return tuple(range(min(pattern.effective_banks_override, bank_model.bank_count)))
    if pattern.region_name is None or pattern.region_name not in regions:
        return tuple(range(bank_model.bank_count))
    region = regions[pattern.region_name]
    banks: Set[int] = set()
    core_samples = max(1, active_cores if region.per_core_stride_bytes else 1)
    samples = max(1, ceil(region.size_bytes / pattern.access_stride_bytes))
    for core_id in range(core_samples):
        base = region.base_offset + core_id * region.per_core_stride_bytes
        for index in range(samples):
            offset = index * pattern.access_stride_bytes
            if offset >= region.size_bytes:
                break
            banks.update(l1_banks_touched(base + offset, pattern.access_width_bytes, bank_model))
            if len(banks) == bank_model.bank_count:
                return tuple(sorted(banks))
    return tuple(sorted(banks))


def expected_served_independent(requesters: int, banks: int, ports_per_bank: int) -> float:
    if requesters <= 0:
        return 0.0
    if banks <= 0:
        raise ValueError('banks must be positive')
    if ports_per_bank == 1:
        return banks * (1.0 - (1.0 - 1.0 / banks) ** requesters)
    from math import comb
    p = 1.0 / banks
    served_per_bank = 0.0
    for occupancy in range(requesters + 1):
        probability = comb(requesters, occupancy) * p ** occupancy * (1.0 - p) ** (requesters - occupancy)
        served_per_bank += min(occupancy, ports_per_bank) * probability
    return banks * served_per_bank


def estimate_bank_pattern(pattern: BankAccessPattern, component_cycles: float, regions: Mapping[str, L1RegionSpec], bank_model: L1BankModel, execution: ExecutionConfig, active_cores: int, pessimism: PessimismConfig) -> Dict[str, Any]:
    requesters = _resolve_requesters(pattern, execution, active_cores)
    banks_tuple = _banks_for_pattern(pattern, regions, bank_model, active_cores)
    expected_banks = max(1, len(banks_tuple))
    pessimistic_banks = max(1, floor(expected_banks * (1.0 - pessimism.bank_spread_haircut)))
    expected_correlation = pattern.correlation
    pessimistic_correlation = min(1.0, pattern.correlation + (1.0 - pattern.correlation) * pessimism.bank_correlation_inflation)
    broadcast = bank_model.same_address_broadcast
    expected_broadcast = pessimism.assume_broadcast_for_expected if broadcast is None else broadcast
    pessimistic_broadcast = pessimism.assume_broadcast_for_pessimistic if broadcast is None else broadcast

    def slowdown(banks: int, correlation: float, broadcast_enabled: bool, conservative: bool) -> Tuple[float, float]:
        if requesters <= 1 or component_cycles <= 0:
            return (1.0, float(requesters))
        independent = expected_served_independent(requesters, banks, bank_model.accesses_per_bank_per_cycle)
        if pattern.broadcast_eligible and broadcast_enabled:
            correlated = float(requesters)
        else:
            correlated = float(min(requesters, bank_model.accesses_per_bank_per_cycle))
        served = (1.0 - correlation) * independent + correlation * correlated
        served = max(1e-12, min(float(requesters), served))
        factor = max(1.0, requesters / served)
        if conservative:
            factor = 1.0 + (factor - 1.0) * pessimism.conflict_excess_factor
            factor = min(float(requesters), factor)
        return (factor, served)
    expected_slowdown, expected_served = slowdown(expected_banks, expected_correlation, bool(expected_broadcast), False)
    pessimistic_slowdown, pessimistic_served = slowdown(pessimistic_banks, pessimistic_correlation, bool(pessimistic_broadcast), True)
    expected_adjusted = component_cycles * expected_slowdown
    pessimistic_adjusted = component_cycles * pessimistic_slowdown
    return {'name': pattern.name, 'component': pattern.component, 'base_component_cycles': component_cycles, 'requesters': requesters, 'banks': banks_tuple, 'effective_banks_expected': expected_banks, 'effective_banks_pessimistic': pessimistic_banks, 'correlation_expected': expected_correlation, 'correlation_pessimistic': pessimistic_correlation, 'broadcast_expected': expected_broadcast, 'broadcast_pessimistic': pessimistic_broadcast, 'served_expected': expected_served, 'served_pessimistic': pessimistic_served, 'slowdown_expected': expected_slowdown, 'slowdown_pessimistic': pessimistic_slowdown, 'adjusted_cycles_expected': expected_adjusted, 'adjusted_cycles_pessimistic': pessimistic_adjusted, 'penalty_cycles_expected': expected_adjusted - component_cycles, 'penalty_cycles_pessimistic': pessimistic_adjusted - component_cycles}


def apply_bank_penalties(components: Mapping[str, float], patterns: Sequence[BankAccessPattern], regions: Mapping[str, L1RegionSpec], bank_model: L1BankModel, execution: ExecutionConfig, active_cores: int, pessimism: PessimismConfig) -> Dict[str, Any]:
    """
    Apply at most one dominant slowdown to each memory component.

    Taking the maximum rather than multiplying patterns avoids counting the
    same stalled cycle several times. Users should define one representative
    pattern per component whenever possible.
    """
    pattern_results: List[Dict[str, Any]] = []
    by_component: Dict[str, List[Dict[str, Any]]] = {}
    for pattern in patterns:
        base = float(components.get(pattern.component, 0.0))
        result = estimate_bank_pattern(pattern, base, regions, bank_model, execution, active_cores, pessimism)
        pattern_results.append(result)
        by_component.setdefault(pattern.component, []).append(result)
    adjusted_expected = dict(components)
    adjusted_pessimistic = dict(components)
    for component, results in by_component.items():
        adjusted_expected[component] = max((result['adjusted_cycles_expected'] for result in results))
        adjusted_pessimistic[component] = max((result['adjusted_cycles_pessimistic'] for result in results))
    expected_total = sum((value for key, value in adjusted_expected.items() if key != 'base_total'))
    pessimistic_total = sum((value for key, value in adjusted_pessimistic.items() if key != 'base_total'))
    return {'patterns': pattern_results, 'components_expected': adjusted_expected, 'components_pessimistic': adjusted_pessimistic, 'cycles_expected_before_safety': expected_total, 'cycles_pessimistic_before_safety': pessimistic_total, 'bank_penalty_expected': expected_total - float(components['base_total']), 'bank_penalty_pessimistic': pessimistic_total - float(components['base_total'])}
