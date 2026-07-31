from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
import csv
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .config import AutoSpecConfig, ExecutionConfig, KernelCostModel, PessimismConfig, TilingModelConfig
from .descriptors import PartitionMode
from .engine import process


@dataclass
class FamilyCalibrationModel:
    family: str
    feature_names: List[str]
    coefficients: List[float]
    safety_factor: float
    sample_count: int
    fit_kind: str

    def predict_base(self, features: Mapping[str, float]) -> float:
        values = np.asarray(
            [1.0] + [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=float,
        )
        coefficients = np.asarray(self.coefficients, dtype=float)
        return float(values @ coefficients)


@dataclass
class FamilyCalibrator:
    models: Dict[str, FamilyCalibrationModel]
    conservative_quantile: float = 0.90
    minimum_safety_factor: float = 1.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conservative_quantile": self.conservative_quantile,
            "minimum_safety_factor": self.minimum_safety_factor,
            "models": {
                name: asdict(model)
                for name, model in self.models.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FamilyCalibrator":
        models = {
            name: FamilyCalibrationModel(**model_data)
            for name, model_data in data.get("models", {}).items()
        }
        return cls(
            models=models,
            conservative_quantile=float(data.get("conservative_quantile", 0.90)),
            minimum_safety_factor=float(data.get("minimum_safety_factor", 1.05)),
        )

    def save(self, path: Any) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Any) -> "FamilyCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def load_measurements_csv(path: Any) -> Dict[str, int]:
    """Read a CSV containing at least layer_name and num_cycles columns."""

    measurements: Dict[str, int] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"layer_name", "num_cycles"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError("measurement CSV is missing columns: %s" % sorted(missing))

        for row in reader:
            name = str(row["layer_name"]).strip()
            if not name:
                continue
            measurements[name] = int(float(row["num_cycles"]))

    return measurements


def classify_node(node: Any, result: Optional[Mapping[str, Any]] = None) -> str:
    """Classify a node without requiring per-node configuration."""

    op_type = str(getattr(node, "op_type", "")).lower()
    name = str(getattr(node, "name", "")).lower()

    kernel_name = ""
    if result is not None:
        automatic = result.get("automatic_metadata", {})
        kernel_name = str(automatic.get("kernel_name") or "").lower()
        if not kernel_name:
            kernel_name = str(result.get("calibration_key") or "").lower()

    if "pool" in op_type or "pool" in name or "pool" in kernel_name:
        return "pooling"

    if (
        "fullyconnected" in op_type
        or "fully_connected" in op_type
        or "gemm" in op_type
        or "linear" in op_type
        or "fullyconnected" in name
        or "linear" in kernel_name
    ):
        return "fully_connected"

    if (
        'pulp_nn_add_' in kernel_name
        or op_type in ('add', 'sum', 'addition', 'reluadd', 'reluaddition')
        or 'addition' in op_type
        or name.startswith('add')
        or 'reluadd' in name
        or 'addition' in name
    ):
        return 'elementwise_add'

    if "conv" in op_type or "conv" in name or "conv" in kernel_name or "depthwise" in kernel_name:
        group = int(getattr(node, "group", 1) or 1)
        input_channels = int(getattr(node, "input_channels", 0) or 0)
        output_channels = int(getattr(node, "output_channels", 0) or 0)

        if "depthwise" in kernel_name:
            return "depthwise_conv"

        if group > 1:
            if input_channels > 0 and group == input_channels:
                return "depthwise_conv"
            return "grouped_conv"

        # A robust fallback when group metadata was lost: compare the graph MAC
        # count with the dense-convolution MAC count.
        try:
            output_dimensions = list(getattr(node, "output_dimensions"))
            kernel_shape = list(getattr(node, "kernel_shape"))
            graph_macs = int(getattr(node, "MACs"))
            dense_macs = (
                int(output_dimensions[0])
                * int(output_dimensions[1])
                * output_channels
                * input_channels
                * int(kernel_shape[0])
                * int(kernel_shape[1])
            )
            if graph_macs > 0 and dense_macs >= 2 * graph_macs:
                inferred_groups = dense_macs / float(graph_macs)
                if inferred_groups >= 2.0:
                    return "depthwise_conv" if inferred_groups >= input_channels * 0.75 else "grouped_conv"
        except (AttributeError, TypeError, ValueError, IndexError):
            pass

        return "standard_conv"

    return "generic"


def _numeric_memory_sum(level: Mapping[str, Any]) -> float:
    preferred_keys = (
        "weight_memory",
        "bias_memory",
        "constants_memory",
        "input_activation_memory",
        "output_activation_memory",
        "lut_memory",
    )

    total = 0.0
    found = False
    for key in preferred_keys:
        value = level.get(key)
        if isinstance(value, (int, float, np.number)):
            total += float(value)
            found = True

    if found:
        return total

    # Conservative fallback for alternate DORY dictionaries.
    for key, value in level.items():
        if key.endswith("_memory") and isinstance(value, (int, float, np.number)):
            total += float(value)
    return total


def memory_footprints(node: Any) -> Tuple[float, float]:
    tiling = getattr(node, "tiling_dimensions", {}) or {}
    l1 = _numeric_memory_sum(tiling.get("L1", {}) or {})
    l2 = _numeric_memory_sum(tiling.get("L2", {}) or {})

    if l1 <= 0:
        l1 = sum(
            float(getattr(node, name, 0) or 0)
            for name in (
                "weight_memory",
                "bias_memory",
                "constants_memory",
                "input_activation_memory",
                "output_activation_memory",
            )
        )
    if l2 <= 0:
        l2 = l1

    return l1, l2


def extract_features(
    node: Any,
    result: Mapping[str, Any],
    family: Optional[str] = None,
) -> Dict[str, float]:
    family = family or classify_node(node, result)
    l1_bytes, l2_bytes = memory_footprints(node)
    strides = list(getattr(node, "strides", [1, 1]) or [1, 1])
    stride_downsample = float(any(int(value) > 1 for value in strides))

    return {
        "raw_pessimistic": float(result["pessimistic_cycles"]),
        "raw_expected": float(result.get("expected_cycles", result["pessimistic_cycles"])),
        "lower_bound": float(result.get("lower_bound_cycles", 0)),
        "macs": float(getattr(node, "MACs", result.get("macs", 0)) or 0),
        "l1_bytes": l1_bytes,
        "l2_bytes": l2_bytes,
        "tiling_excess_bytes": max(0.0, l2_bytes - l1_bytes),
        "stride_downsample": stride_downsample,
        "num_cores": float(result.get("num_cores", 1)),
    }


def _feature_names_for_family(family: str) -> List[str]:
    if family == "standard_conv":
        # The analytical model is already structurally useful for standard
        # convolutions. Correct its scale and add an explicit L2-tiling term.
        return ["raw_pessimistic", "tiling_excess_bytes"]

    if family in ("depthwise_conv", "grouped_conv"):
        # The standard source model is structurally wrong for these kernels.
        # Use graph MACs, memory footprint, and downsampling instead of its raw
        # source-operation count.
        return ["macs", "l1_bytes", "stride_downsample"]

    # Pooling, fully connected, and uncommon operators often have too few
    # samples for a multi-feature regression. Use a stable affine raw model.
    return ["raw_pessimistic"]


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Fit an affine ridge model; the intercept is not regularized."""

    regularizer = np.eye(x.shape[1], dtype=float) * float(ridge)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def fit_family_calibrator(
    graph: Sequence[Any],
    raw_results: Sequence[Mapping[str, Any]],
    measured_cycles: Mapping[str, int],
    conservative_quantile: float = 0.90,
    minimum_safety_factor: float = 1.05,
    ridge: float = 1e-8,
) -> FamilyCalibrator:
    """Fit operation-family corrections from measured GVSoC cycles.

    For reliable generalization, fit on representative layers or networks and
    validate on held-out layers/networks.  Fitting and evaluating on the same
    graph only measures in-sample fit.
    """

    if not 0.5 <= conservative_quantile <= 1.0:
        raise ValueError("conservative_quantile must be in [0.5, 1]")
    if minimum_safety_factor < 1.0:
        raise ValueError("minimum_safety_factor must be >= 1")
    if len(graph) != len(raw_results):
        raise ValueError("graph and raw_results must have the same length")

    rows_by_family: Dict[str, List[Tuple[Dict[str, float], float]]] = {}

    for node, result in zip(graph, raw_results):
        name = str(result.get("name", getattr(node, "name", "")))
        if name not in measured_cycles:
            continue

        measured = float(measured_cycles[name])
        if measured <= 0:
            continue

        family = classify_node(node, result)
        features = extract_features(node, result, family)
        rows_by_family.setdefault(family, []).append((features, measured))

    models: Dict[str, FamilyCalibrationModel] = {}

    for family, rows in rows_by_family.items():
        feature_names = _feature_names_for_family(family)
        x = np.asarray(
            [
                [1.0] + [features[name] for name in feature_names]
                for features, _ in rows
            ],
            dtype=float,
        )
        y = np.asarray([measured for _, measured in rows], dtype=float)

        # With very few samples, a multi-parameter fit is underdetermined.
        # Fall back to a pure multiplicative correction of raw pessimistic.
        if len(rows) < x.shape[1] + 1:
            feature_names = ["raw_pessimistic"]
            raw = np.asarray(
                [max(1.0, features["raw_pessimistic"]) for features, _ in rows],
                dtype=float,
            )
            scale = float(np.median(y / raw))
            coefficients = np.asarray([0.0, scale], dtype=float)
            fitted = raw * scale
            fit_kind = "multiplicative_fallback"
        else:
            coefficients = _fit_ridge(x, y, ridge)
            fitted = x @ coefficients
            fit_kind = "affine_ridge"

        fitted = np.maximum(1.0, fitted)
        ratios = y / fitted
        safety_factor = max(
            float(minimum_safety_factor),
            float(np.quantile(ratios, conservative_quantile)),
        )

        models[family] = FamilyCalibrationModel(
            family=family,
            feature_names=list(feature_names),
            coefficients=[float(value) for value in coefficients],
            safety_factor=float(safety_factor),
            sample_count=len(rows),
            fit_kind=fit_kind,
        )

    return FamilyCalibrator(
        models=models,
        conservative_quantile=conservative_quantile,
        minimum_safety_factor=minimum_safety_factor,
    )


def apply_family_calibrator(
    graph: Sequence[Any],
    raw_results: Sequence[Mapping[str, Any]],
    calibrator: FamilyCalibrator,
) -> List[Dict[str, Any]]:
    """Return shallow result copies with calibrated latency fields added."""

    if len(graph) != len(raw_results):
        raise ValueError("graph and raw_results must have the same length")

    calibrated_results: List[Dict[str, Any]] = []

    for node, raw in zip(graph, raw_results):
        result = dict(raw)
        family = classify_node(node, raw)
        features = extract_features(node, raw, family)
        model = calibrator.models.get(family)

        if model is None:
            base = float(raw["pessimistic_cycles"])
            pessimistic = int(ceil(base * calibrator.minimum_safety_factor))
            source = "uncalibrated_family"
            safety_factor = calibrator.minimum_safety_factor
        else:
            base = model.predict_base(features)
            base = max(
                float(raw.get("lower_bound_cycles", 0)),
                1.0,
                base,
            )
            pessimistic = int(ceil(base * model.safety_factor))
            source = model.fit_kind
            safety_factor = model.safety_factor

        result["family"] = family
        result["calibrated_expected_cycles"] = int(ceil(base))
        result["calibrated_pessimistic_cycles"] = pessimistic
        result["calibration"] = {
            "source": source,
            "safety_factor": float(safety_factor),
            "features": features,
        }
        calibrated_results.append(result)

    return calibrated_results


def get_pessimistic_cycles(result: Mapping[str, Any]) -> int:
    """Return calibrated pessimistic cycles when available, otherwise raw."""

    return int(
        result.get(
            "calibrated_pessimistic_cycles",
            result["pessimistic_cycles"],
        )
    )


def process_calibrated(
    graph: Sequence[Any],
    hw_spec: Mapping[str, Any],
    execution: ExecutionConfig,
    *,
    auto_spec: Optional[AutoSpecConfig] = None,
    measured_cycles: Optional[Mapping[str, int]] = None,
    measurements_csv: Optional[Any] = None,
    calibrator: Optional[FamilyCalibrator] = None,
    calibration_path: Optional[Any] = None,
    fit_calibration: bool = False,
    save_calibration_path: Optional[Any] = None,
    conservative_quantile: float = 0.90,
    minimum_safety_factor: float = 1.05,
    kernel_cost: Optional[KernelCostModel] = None,
    pessimism: Optional[PessimismConfig] = None,
    tiling: Optional[TilingModelConfig] = None,
    peak_key: str = "8bits",
    partition_mode: PartitionMode = "implementation_exact",
    kernel_calibration_scales: Optional[Mapping[str, float]] = None,
    global_calibration_scale: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Optional[FamilyCalibrator]]:
    """Run the analytical model and optionally apply family calibration.

    Calibration choices:

    * Pass ``calibrator`` directly.
    * Pass ``calibration_path`` to load an existing JSON calibration.
    * Set ``fit_calibration=True`` and provide ``measured_cycles`` or
      ``measurements_csv`` to fit a new calibration.

    Returns ``(results, calibrator)``. Each result exposes
    ``calibrated_pessimistic_cycles`` when calibration is active.
    """

    measurements: Dict[str, int] = {}
    if measurements_csv is not None:
        measurements.update(load_measurements_csv(measurements_csv))
    if measured_cycles is not None:
        measurements.update({str(k): int(v) for k, v in measured_cycles.items()})

    raw_results = process(
        graph=graph,
        hw_spec=hw_spec,
        execution=execution,
        auto_spec=auto_spec,
        measured_cycles=measurements or None,
        kernel_cost=kernel_cost,
        pessimism=pessimism,
        tiling=tiling,
        peak_key=peak_key,
        partition_mode=partition_mode,
        kernel_calibration_scales=kernel_calibration_scales,
        global_calibration_scale=global_calibration_scale,
    )

    active_calibrator = calibrator
    if active_calibrator is None and calibration_path is not None:
        active_calibrator = FamilyCalibrator.load(calibration_path)

    if fit_calibration:
        if not measurements:
            raise ValueError(
                "fit_calibration=True requires measured_cycles or measurements_csv"
            )
        active_calibrator = fit_family_calibrator(
            graph=graph,
            raw_results=raw_results,
            measured_cycles=measurements,
            conservative_quantile=conservative_quantile,
            minimum_safety_factor=minimum_safety_factor,
        )

    if active_calibrator is None:
        final_results: List[Dict[str, Any]] = []
        for raw in raw_results:
            item = dict(raw)
            item["family"] = None
            item["calibrated_expected_cycles"] = int(raw["expected_cycles"])
            item["calibrated_pessimistic_cycles"] = int(raw["pessimistic_cycles"])
            final_results.append(item)
    else:
        final_results = apply_family_calibrator(
            graph=graph,
            raw_results=raw_results,
            calibrator=active_calibrator,
        )

    if save_calibration_path is not None:
        if active_calibrator is None:
            raise ValueError(
                "save_calibration_path was provided but no calibrator is active"
            )
        active_calibrator.save(save_calibration_path)

    return final_results, active_calibrator
