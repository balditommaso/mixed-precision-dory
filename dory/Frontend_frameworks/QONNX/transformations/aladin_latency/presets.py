from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping


def prepare_pulp_hw_spec(
    hw_spec: Mapping[str, Any],
    *,
    reference_cores: int = 8,
    dma_bandwidth_bytes_per_cycle: float = 8.0,
    l1_banks: int = 16,
    bank_interleaving_bytes: int = 4,
    inplace: bool = False,
) -> Dict[str, Any]:
    """Apply the model's standard PULP assumptions to a hardware description.

    By default a deep copy is returned, so the hardware description loaded by
    DORY is not modified. Set ``inplace=True`` only when mutation is desired.
    """

    if reference_cores <= 0:
        raise ValueError("reference_cores must be positive")
    if dma_bandwidth_bytes_per_cycle <= 0:
        raise ValueError("DMA bandwidth must be positive")
    if l1_banks <= 0 or bank_interleaving_bytes <= 0:
        raise ValueError("L1 bank parameters must be positive")

    result = hw_spec if inplace else deepcopy(dict(hw_spec))

    result.setdefault("compute_model", {}).update(
        {
            "peak_scope": "cluster",
            "reference_cores": reference_cores,
        }
    )

    result.setdefault("latency_model", {}).setdefault("dma_l2_l1", {}).update(
        {
            "read_bandwidth_bytes_per_cycle": dma_bandwidth_bytes_per_cycle,
            "write_bandwidth_bytes_per_cycle": dma_bandwidth_bytes_per_cycle,
            "transaction_startup_cycles": 10,
            "push_cycles": 2,
            "barrier_call_cycles": 4,
            "bandwidth_efficiency": 0.90,
            "pessimistic_bandwidth_efficiency": 0.80,
            "pessimistic_startup_factor": 1.15,
        }
    )

    result["memory"]["L1"].update(
        {
            "banks": l1_banks,
            "bank_interleaving_bytes": bank_interleaving_bytes,
            "accesses_per_bank_per_cycle": 1,
            "same_address_broadcast": False,
            "dma_shares_bank_ports": True,
            "arbitration": "round_robin",
        }
    )
    return result
