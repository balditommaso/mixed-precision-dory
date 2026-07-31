"""Modular DORY/PULP analytical latency estimator."""

from .calibration import (
    FamilyCalibrationModel,
    FamilyCalibrator,
    apply_family_calibrator,
    classify_node,
    fit_family_calibrator,
    get_pessimistic_cycles,
    load_measurements_csv,
    process_calibrated,
)
from .config import (
    AutoSpecConfig,
    DMAHardwareModel,
    ExecutionConfig,
    KernelCostModel,
    L1BankModel,
    PessimismConfig,
    TilingModelConfig,
)
from .descriptors import (
    AddKernelSpec,
    BankAccessPattern,
    DMAKind,
    DMATransferSpec,
    KernelComputeSpec,
    L1RegionSpec,
    LinearKernelSpec,
    NodeExecutionSpec,
    NodeSourceMetadata,
    PoolKernelSpec,
)
from .engine import derive_calibration_scale, process
from .tiling import TileLoopCounts, TilePlan
from .estimator import LatencyEstimator
from .presets import prepare_pulp_hw_spec
from .reporting import (
    format_diagnostics,
    format_pessimistic,
    format_summary,
    print_pessimistic_latencies,
)

__all__ = [
    "AddKernelSpec",
    "AutoSpecConfig",
    "BankAccessPattern",
    "DMAHardwareModel",
    "DMAKind",
    "DMATransferSpec",
    "ExecutionConfig",
    "FamilyCalibrationModel",
    "FamilyCalibrator",
    "KernelComputeSpec",
    "KernelCostModel",
    "L1BankModel",
    "L1RegionSpec",
    "LatencyEstimator",
    "LinearKernelSpec",
    "NodeExecutionSpec",
    "NodeSourceMetadata",
    "PessimismConfig",
    "TilingModelConfig",
    "TileLoopCounts",
    "TilePlan",
    "PoolKernelSpec",
    "apply_family_calibrator",
    "classify_node",
    "derive_calibration_scale",
    "fit_family_calibrator",
    "format_diagnostics",
    "format_pessimistic",
    "format_summary",
    "get_pessimistic_cycles",
    "load_measurements_csv",
    "prepare_pulp_hw_spec",
    "print_pessimistic_latencies",
    "process",
    "process_calibrated",
]
