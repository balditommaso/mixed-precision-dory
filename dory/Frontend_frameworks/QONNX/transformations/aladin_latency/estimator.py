from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .calibration import FamilyCalibrator, process_calibrated
from .config import AutoSpecConfig, ExecutionConfig, KernelCostModel, PessimismConfig
from .descriptors import PartitionMode
from .engine import process
from .reporting import print_pessimistic_latencies


@dataclass
class LatencyEstimator:
    """Convenient stateful front end for the analytical latency model.

    The existing functional ``process`` API remains available. This class keeps
    the hardware and global configuration in one place, reducing notebook
    boilerplate and making repeated graph evaluations less error-prone.
    """

    hw_spec: Mapping[str, Any]
    execution: ExecutionConfig = field(default_factory=lambda: ExecutionConfig(num_cores=8))
    auto_spec: AutoSpecConfig = field(default_factory=AutoSpecConfig)
    kernel_cost: KernelCostModel = field(default_factory=KernelCostModel)
    pessimism: PessimismConfig = field(default_factory=PessimismConfig)
    peak_key: str = "8bits"
    partition_mode: PartitionMode = "implementation_exact"
    kernel_calibration_scales: Mapping[str, float] = field(default_factory=dict)
    global_calibration_scale: float = 1.0

    @classmethod
    def simple(
        cls,
        hw_spec: Mapping[str, Any],
        *,
        num_cores: int = 8,
        generated_code_dir: Optional[Union[str, Path]] = None,
        parse_generated_code: bool = True,
        pessimism: Optional[PessimismConfig] = None,
        kernel_cost: Optional[KernelCostModel] = None,
    ) -> "LatencyEstimator":
        """Create an estimator using the recommended defaults."""
        return cls(
            hw_spec=hw_spec,
            execution=ExecutionConfig(num_cores=num_cores),
            auto_spec=AutoSpecConfig(
                generated_code_dir=generated_code_dir,
                parse_generated_code=(
                    parse_generated_code and generated_code_dir is not None
                ),
            ),
            pessimism=pessimism or PessimismConfig(),
            kernel_cost=kernel_cost or KernelCostModel(),
        )

    def process(
        self,
        graph: Sequence[Any],
        *,
        measured_cycles: Optional[Mapping[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        return process(
            graph=graph,
            hw_spec=self.hw_spec,
            execution=self.execution,
            auto_spec=self.auto_spec,
            measured_cycles=measured_cycles,
            kernel_cost=self.kernel_cost,
            pessimism=self.pessimism,
            peak_key=self.peak_key,
            partition_mode=self.partition_mode,
            kernel_calibration_scales=self.kernel_calibration_scales,
            global_calibration_scale=self.global_calibration_scale,
        )

    def process_calibrated(
        self,
        graph: Sequence[Any],
        *,
        measured_cycles: Optional[Mapping[str, int]] = None,
        measurements_csv: Optional[Any] = None,
        calibrator: Optional[FamilyCalibrator] = None,
        calibration_path: Optional[Any] = None,
        fit_calibration: bool = False,
        save_calibration_path: Optional[Any] = None,
        conservative_quantile: float = 0.90,
        minimum_safety_factor: float = 1.05,
    ):
        return process_calibrated(
            graph=graph,
            hw_spec=self.hw_spec,
            execution=self.execution,
            auto_spec=self.auto_spec,
            measured_cycles=measured_cycles,
            measurements_csv=measurements_csv,
            calibrator=calibrator,
            calibration_path=calibration_path,
            fit_calibration=fit_calibration,
            save_calibration_path=save_calibration_path,
            conservative_quantile=conservative_quantile,
            minimum_safety_factor=minimum_safety_factor,
            kernel_cost=self.kernel_cost,
            pessimism=self.pessimism,
            peak_key=self.peak_key,
            partition_mode=self.partition_mode,
            kernel_calibration_scales=self.kernel_calibration_scales,
            global_calibration_scale=self.global_calibration_scale,
        )

    @staticmethod
    def print(
        results,
        *,
        include_total: bool = False,
        frequency_hz: Optional[float] = None,
    ) -> None:
        print_pessimistic_latencies(
            results,
            include_total=include_total,
            frequency_hz=frequency_hz,
        )
