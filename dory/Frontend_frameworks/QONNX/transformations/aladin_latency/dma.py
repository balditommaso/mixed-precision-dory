from __future__ import annotations

from math import ceil, log2
from typing import Any, Dict, List

from .config import DMAHardwareModel, ExecutionConfig
from .descriptors import DMAKind, DMATransferSpec


def select_dma_kind(transfer: DMATransferSpec) -> DMAKind:
    """Mirror the supplied dory_dma_memcpy_async dispatcher."""
    n2 = transfer.number_of_2d_copies
    n1 = transfer.number_of_1d_copies
    length = transfer.length_1d_copy
    stride_1d = transfer.stride_1d
    stride_2d = transfer.stride_2d
    if transfer.hwc_to_chw:
        return DMAKind.HWC_TO_CHW
    if n2 == 1 and n1 == 1 or (stride_1d == length and n1 * length == stride_2d) or (n2 == 1 and length == stride_1d):
        return DMAKind.ONE_D
    if n2 == 1 or length == stride_1d:
        return DMAKind.TWO_D
    return DMAKind.THREE_D


def implementation_partition_counts(work_items: int, num_cores: int) -> List[int]:
    """Mirror the DORY shift-and-mask partitioning used by 3D/HWC DMA."""
    if work_items <= 0:
        return [0] * num_cores
    log2_cores = int(log2(num_cores))
    chunk = (work_items >> log2_cores) + int(work_items & num_cores - 1 != 0)
    counts: List[int] = []
    for core_id in range(num_cores):
        start = min(chunk * core_id, work_items)
        stop = min(start + chunk, work_items)
        counts.append(max(0, stop - start))
    return counts


def mchan_commands_per_core(transfer: DMATransferSpec, execution: ExecutionConfig) -> List[int]:
    kind = select_dma_kind(transfer)
    if kind in (DMAKind.ONE_D, DMAKind.TWO_D):
        result = [0] * execution.num_cores
        result[0] = transfer.submissions
        return result
    work_items = transfer.number_of_2d_copies if kind == DMAKind.THREE_D else transfer.length_1d_copy
    if execution.single_core_dma:
        result = [0] * execution.num_cores
        result[0] = work_items * transfer.submissions
        return result
    return [count * transfer.submissions for count in implementation_partition_counts(work_items, execution.num_cores)]


def estimate_dma_transfer(transfer: DMATransferSpec, execution: ExecutionConfig, dma_hw: DMAHardwareModel) -> Dict[str, Any]:
    kind = select_dma_kind(transfer)
    bandwidth = dma_hw.read_bandwidth_bytes_per_cycle if transfer.direction == 'L2_TO_L1' else dma_hw.write_bandwidth_bytes_per_cycle
    commands_per_core = mchan_commands_per_core(transfer, execution)
    command_count = sum(commands_per_core)
    max_pushes = max(commands_per_core, default=0)
    expected_payload = ceil(transfer.physical_bytes / (bandwidth * dma_hw.bandwidth_efficiency))
    pessimistic_payload = ceil(transfer.physical_bytes / (bandwidth * dma_hw.pessimistic_bandwidth_efficiency))
    expected_startup = command_count * dma_hw.transaction_startup_cycles
    pessimistic_startup = ceil(expected_startup * dma_hw.pessimistic_startup_factor)
    issue_cycles = max_pushes * dma_hw.push_cycles
    internal_barriers = 0
    if execution.always_block_dma_transfers and kind in (DMAKind.THREE_D, DMAKind.HWC_TO_CHW):
        internal_barriers = max_pushes
    barrier_cycles = (transfer.barrier_calls + internal_barriers) * dma_hw.barrier_call_cycles
    expected_cycles = expected_payload + expected_startup + issue_cycles + barrier_cycles
    pessimistic_cycles = pessimistic_payload + pessimistic_startup + issue_cycles + barrier_cycles
    logical_difference = None
    if transfer.logical_bytes is not None:
        logical_difference = transfer.physical_bytes - transfer.logical_bytes
    return {'name': transfer.name, 'kind': kind.value, 'direction': transfer.direction, 'logical_bytes': transfer.logical_bytes, 'physical_bytes': transfer.physical_bytes, 'logical_physical_difference': logical_difference, 'mchan_transactions': command_count, 'commands_per_core': commands_per_core, 'payload_cycles_expected': expected_payload, 'payload_cycles_pessimistic': pessimistic_payload, 'startup_cycles_expected': expected_startup, 'startup_cycles_pessimistic': pessimistic_startup, 'issue_cycles': issue_cycles, 'barrier_cycles': barrier_cycles, 'expected_cycles': expected_cycles, 'pessimistic_cycles': pessimistic_cycles}
