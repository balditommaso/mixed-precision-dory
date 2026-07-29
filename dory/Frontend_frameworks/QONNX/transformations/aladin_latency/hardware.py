from __future__ import annotations

from typing import Any, Mapping

from .config import DMAHardwareModel, L1BankModel
from .descriptors import KernelComputeSpec


def get_dma_hardware_model(hw_spec: Mapping[str, Any]) -> DMAHardwareModel:
    l2 = hw_spec['memory']['L2']
    config = hw_spec.get('latency_model', {}).get('dma_l2_l1', {})
    return DMAHardwareModel(
        read_bandwidth_bytes_per_cycle=float(config.get('read_bandwidth_bytes_per_cycle', l2['bandwidth'])), 
        write_bandwidth_bytes_per_cycle=float(config.get('write_bandwidth_bytes_per_cycle', l2['bandwidth'])), 
        transaction_startup_cycles=int(config.get('transaction_startup_cycles', l2.get('latency', 0) or 0)), 
        push_cycles=int(config.get('push_cycles', 2)), barrier_call_cycles=int(config.get('barrier_call_cycles', 4)), 
        bandwidth_efficiency=float(config.get('bandwidth_efficiency', 0.9)), 
        pessimistic_bandwidth_efficiency=float(config.get('pessimistic_bandwidth_efficiency', 0.8)), 
        pessimistic_startup_factor=float(config.get('pessimistic_startup_factor', 1.15))
    )


def get_l1_bank_model(hw_spec: Mapping[str, Any]) -> L1BankModel:
    l1 = hw_spec['memory']['L1']
    return L1BankModel(
        bank_count=int(l1.get('banks', 16)), 
        interleaving_bytes=int(l1.get('bank_interleaving_bytes', 4)), 
        accesses_per_bank_per_cycle=int(l1.get('accesses_per_bank_per_cycle', 1)), 
        same_address_broadcast=l1.get('same_address_broadcast'), 
        dma_shares_bank_ports=bool(l1.get('dma_shares_bank_ports', True)), 
        arbitration=str(l1.get('arbitration', 'round_robin'))
    )


def get_peak_mac_per_cycle_per_core(hw_spec: Mapping[str, Any], peak_key: str, kernel: KernelComputeSpec) -> float:
    if kernel.peak_mac_per_cycle_per_core is not None:
        return kernel.peak_mac_per_cycle_per_core
    peak = float(hw_spec['peak MAC/cycle'][peak_key])
    compute_model = hw_spec.get('compute_model', {})
    scope = str(compute_model.get('peak_scope', 'cluster'))
    if scope == 'core':
        return peak
    if scope == 'cluster':
        reference_cores = int(compute_model.get('reference_cores', 8))
        if reference_cores <= 0:
            raise ValueError('reference_cores must be positive')
        return peak / reference_cores
    raise ValueError(f'unsupported peak scope: {scope}')
