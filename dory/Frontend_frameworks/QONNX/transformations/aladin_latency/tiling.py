from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .config import TilingModelConfig
from .descriptors import (
    AddKernelSpec,
    ComputeKernelSpec,
    KernelComputeSpec,
    L1RegionSpec,
    LinearKernelSpec,
    NodeExecutionSpec,
    PoolKernelSpec,
)


_ROLE_MEMORY_KEYS = {
    "input": ("input_activation_memory", "input_memory"),
    "second_input": (
        "second_input_activation_memory",
        "input2_activation_memory",
        "second_input_memory",
    ),
    "output": ("output_activation_memory", "output_memory"),
    "weights": ("weight_memory", "weights_memory"),
    "bias": ("bias_memory",),
    "constants": ("constants_memory", "constant_memory"),
}


@dataclass(frozen=True)
class TileLoopCounts:
    """Runtime loop counts for one DORY tiling level.

    DORY stores activation dimensions in CHW order and weight dimensions in
    [output_channels, input_channels] order.  These counts describe why a
    layer is tiled, rather than collapsing the schedule into one ambiguous
    ``total_tiles`` scalar.
    """

    output_height: int = 1
    output_width: int = 1
    output_channels: int = 1
    input_channels: int = 1
    channels_coupled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "output_height",
            "output_width",
            "output_channels",
            "input_channels",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} tile count must be positive")

    @property
    def spatial_tiles(self) -> int:
        return self.output_height * self.output_width

    @property
    def output_tiles(self) -> int:
        return self.spatial_tiles * self.output_channels

    @property
    def kernel_calls(self) -> int:
        if self.channels_coupled:
            return self.output_tiles
        return self.output_tiles * self.input_channels

    def repetitions_for_role(self, role: str) -> int:
        """Approximate transfer repetitions from the dimensional loop nest.

        The policy assumes input activation reuse across output-channel tiles
        and weight reuse across spatial tiles.  It avoids the previous error
        of repeating every tensor ``total_tiles`` times.
        """

        role = str(role)
        if role in ("input", "second_input"):
            channel_tiles = (
                self.output_channels
                if self.channels_coupled
                else self.input_channels
            )
            return max(1, self.spatial_tiles * channel_tiles)
        if role == "weights":
            reduction_tiles = 1 if self.channels_coupled else self.input_channels
            return max(1, self.output_channels * reduction_tiles)
        if role in ("bias", "constants"):
            return max(1, self.output_channels)
        if role == "output":
            return max(1, self.output_tiles)
        return max(1, self.kernel_calls)


@dataclass(frozen=True)
class TileGroup:
    """One unique tile shape and the number of times it is executed."""

    kernel: ComputeKernelSpec
    multiplicity: int
    role_bytes: Mapping[str, int]
    shape: Mapping[str, int]


@dataclass(frozen=True)
class TilePlan:
    enabled: bool
    source: str
    parsed_total_tiles: int
    derived_total_tiles: int
    runtime_total_tiles: int
    groups: Tuple[TileGroup, ...]
    warnings: Tuple[str, ...] = ()
    loop_counts: TileLoopCounts = TileLoopCounts()
    inner_loop_counts: TileLoopCounts = TileLoopCounts()
    outer_loop_counts: TileLoopCounts = TileLoopCounts()
    level_name: str = "L1"
    parent_level_name: str = "L2"

    @property
    def has_explicit_shapes(self) -> bool:
        return bool(self.groups)


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default



def _level(node: Any, level_name: str) -> Mapping[str, Any]:
    tiling = getattr(node, "tiling_dimensions", {}) or {}
    value = tiling.get(level_name, {}) if isinstance(tiling, Mapping) else {}
    return value if isinstance(value, Mapping) else {}



def _dims(level: Mapping[str, Any], role: str) -> Tuple[int, ...]:
    candidates = (
        f"{role}_dimensions",
        f"{role}_dimension",
        f"{role}_shape",
    )
    for key in candidates:
        raw = level.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = tuple(_positive_int(item) for item in raw)
            values = tuple(item for item in values if item > 0)
            if values:
                return values
    return ()



def _channel(level: Mapping[str, Any], role: str, fallback: int) -> int:
    """Return the channel dimension using DORY's exact CHW convention."""

    dims = _dims(level, role)
    if len(dims) >= 1:
        return min(max(1, int(fallback)), dims[0])
    return max(1, int(fallback))



def _spatial(
    level: Mapping[str, Any],
    role: str,
    fallback_h: int,
    fallback_w: int,
) -> Tuple[int, int]:
    """Return H and W from DORY activation dimensions [C, H, W]."""

    dims = _dims(level, role)
    if len(dims) >= 3:
        return (
            min(max(1, int(fallback_h)), dims[1]),
            min(max(1, int(fallback_w)), dims[2]),
        )
    return max(1, int(fallback_h)), max(1, int(fallback_w))



def _weight_channels(
    level: Mapping[str, Any],
    fallback_output_channels: int,
    fallback_input_channels: int,
) -> Tuple[int, int]:
    """Return DORY weight tile dimensions [C_out, C_in]."""

    dims = _dims(level, "weights")
    if len(dims) >= 2:
        return (
            min(max(1, int(fallback_output_channels)), dims[0]),
            min(max(1, int(fallback_input_channels)), dims[1]),
        )
    return (
        max(1, int(fallback_output_channels)),
        max(1, int(fallback_input_channels)),
    )



def _next_level_name(level_name: str) -> str:
    """Return the parent memory level, e.g. L1 -> L2."""

    name = str(level_name).upper()
    if name.startswith("L") and name[1:].isdigit():
        return f"L{int(name[1:]) + 1}"
    return "L2"



def _ceil_div(total: int, tile: int) -> int:
    return int(ceil(max(1, int(total)) / float(max(1, int(tile)))))



def _combine_loop_counts(
    outer: TileLoopCounts,
    inner: TileLoopCounts,
) -> TileLoopCounts:
    return TileLoopCounts(
        output_height=outer.output_height * inner.output_height,
        output_width=outer.output_width * inner.output_width,
        output_channels=outer.output_channels * inner.output_channels,
        input_channels=outer.input_channels * inner.input_channels,
        channels_coupled=outer.channels_coupled or inner.channels_coupled,
    )



def _conv_loop_counts(
    node: Any,
    kernel: KernelComputeSpec,
    child: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> Tuple[TileLoopCounts, TileLoopCounts, TileLoopCounts, List[str]]:
    """Derive full->parent and parent->child convolution loop counts."""

    warnings: List[str] = []

    parent_out_h, parent_out_w = _spatial(
        parent, "output", kernel.output_height, kernel.output_width
    )
    child_out_h, child_out_w = _spatial(
        child, "output", parent_out_h, parent_out_w
    )

    parent_out_c_activation = _channel(
        parent, "output", kernel.output_channels
    )
    parent_in_c_activation = _channel(
        parent, "input", kernel.input_channels
    )
    child_out_c_activation = _channel(
        child, "output", parent_out_c_activation
    )
    child_in_c_activation = _channel(
        child, "input", parent_in_c_activation
    )

    parent_out_c, parent_in_c = _weight_channels(
        parent,
        parent_out_c_activation,
        parent_in_c_activation,
    )
    child_out_c, child_in_c = _weight_channels(
        child,
        child_out_c_activation,
        child_in_c_activation,
    )

    # Activation and weight descriptors should agree. Prefer weights_dimensions
    # for convolution channel tiling because it is generated directly by DORY's
    # tiler as [C_out, C_in].
    if parent_out_c != parent_out_c_activation or child_out_c != child_out_c_activation:
        warnings.append(
            "Activation and weight output-channel tile sizes differ; "
            "weights_dimensions[0] is used for convolution tiling."
        )
    if parent_in_c != parent_in_c_activation or child_in_c != child_in_c_activation:
        warnings.append(
            "Activation and weight input-channel tile sizes differ; "
            "weights_dimensions[1] is used for convolution tiling."
        )

    coupled = bool(kernel.is_depthwise)
    if coupled:
        # In depthwise convolution, input- and output-channel tiles are the same
        # runtime loop. Counting both would square the number of channel tiles.
        parent_in_c = kernel.input_channels
        child_in_c = parent_in_c

    inner = TileLoopCounts(
        output_height=_ceil_div(parent_out_h, child_out_h),
        output_width=_ceil_div(parent_out_w, child_out_w),
        output_channels=_ceil_div(parent_out_c, child_out_c),
        input_channels=(1 if coupled else _ceil_div(parent_in_c, child_in_c)),
        channels_coupled=coupled,
    )
    outer = TileLoopCounts(
        output_height=_ceil_div(kernel.output_height, parent_out_h),
        output_width=_ceil_div(kernel.output_width, parent_out_w),
        output_channels=_ceil_div(kernel.output_channels, parent_out_c),
        input_channels=(
            1
            if coupled
            else _ceil_div(kernel.input_channels, parent_in_c)
        ),
        channels_coupled=coupled,
    )
    combined = _combine_loop_counts(outer, inner)

    if kernel.is_grouped and not kernel.is_depthwise and combined.input_channels > 1:
        warnings.append(
            "Grouped-convolution reduction-channel tiling requires exact group "
            "mapping; the loop count is reported but compute reconstruction "
            "should remain in full_layer mode."
        )

    return combined, inner, outer, warnings



def _activation_loop_counts(
    kernel: ComputeKernelSpec,
    child: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> Tuple[TileLoopCounts, TileLoopCounts, TileLoopCounts, List[str]]:
    """Derive loops for pooling and elementwise operators.

    Their channel tiles are coupled between input and output and therefore form
    one channel loop rather than independent input/output channel loops.
    """

    if isinstance(kernel, PoolKernelSpec):
        full_h, full_w, full_c = (
            kernel.output_height,
            kernel.output_width,
            kernel.channels,
        )
    elif isinstance(kernel, AddKernelSpec):
        full_h, full_w, full_c = kernel.height, kernel.width, kernel.channels
    else:
        raise TypeError(type(kernel))

    parent_h, parent_w = _spatial(parent, "output", full_h, full_w)
    child_h, child_w = _spatial(child, "output", parent_h, parent_w)
    parent_c = _channel(parent, "output", full_c)
    child_c = _channel(child, "output", parent_c)

    inner = TileLoopCounts(
        output_height=_ceil_div(parent_h, child_h),
        output_width=_ceil_div(parent_w, child_w),
        output_channels=_ceil_div(parent_c, child_c),
        input_channels=1,
        channels_coupled=True,
    )
    outer = TileLoopCounts(
        output_height=_ceil_div(full_h, parent_h),
        output_width=_ceil_div(full_w, parent_w),
        output_channels=_ceil_div(full_c, parent_c),
        input_channels=1,
        channels_coupled=True,
    )
    return _combine_loop_counts(outer, inner), inner, outer, []



def _linear_loop_counts(
    kernel: LinearKernelSpec,
    child: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> Tuple[TileLoopCounts, TileLoopCounts, TileLoopCounts, List[str]]:
    warnings: List[str] = []

    parent_out, parent_in = _weight_channels(
        parent, kernel.output_neurons, kernel.input_features
    )
    child_out, child_in = _weight_channels(child, parent_out, parent_in)

    inner = TileLoopCounts(
        output_channels=_ceil_div(parent_out, child_out),
        input_channels=_ceil_div(parent_in, child_in),
    )
    outer = TileLoopCounts(
        output_channels=_ceil_div(kernel.output_neurons, parent_out),
        input_channels=_ceil_div(kernel.input_features, parent_in),
    )
    combined = _combine_loop_counts(outer, inner)

    if combined.input_channels > 1:
        warnings.append(
            "Fully-connected reduction tiling is present in weights_dimensions[1]. "
            "The supplied linear kernel has no partial-accumulation interface; "
            "keep compute_mode='full_layer' unless the generated wrapper is parsed."
        )

    return combined, inner, outer, warnings



def derive_loop_counts(
    node: Any,
    kernel: ComputeKernelSpec,
    level_name: str,
) -> Tuple[TileLoopCounts, TileLoopCounts, TileLoopCounts, str, List[str]]:
    """Derive dimensional runtime loops from exact HW_node fields.

    ``HW_node`` stores activations as [C, H, W] and weights as [C_out, C_in].
    The returned values are respectively the combined full->L1 loops, the
    parent->L1 loops, and the full->parent loops.
    """

    child = _level(node, level_name)
    parent_name = _next_level_name(level_name)
    parent = _level(node, parent_name)

    # When no parent level exists, the kernel dimensions represent the parent.
    if isinstance(kernel, KernelComputeSpec):
        combined, inner, outer, warnings = _conv_loop_counts(
            node, kernel, child, parent
        )
        return combined, inner, outer, parent_name, warnings
    if isinstance(kernel, (PoolKernelSpec, AddKernelSpec)):
        combined, inner, outer, warnings = _activation_loop_counts(
            kernel, child, parent
        )
        return combined, inner, outer, parent_name, warnings
    if isinstance(kernel, LinearKernelSpec):
        combined, inner, outer, warnings = _linear_loop_counts(
            kernel, child, parent
        )
        return combined, inner, outer, parent_name, warnings
    return TileLoopCounts(), TileLoopCounts(), TileLoopCounts(), parent_name, []


def _memory(level: Mapping[str, Any], role: str, fallback: int = 0) -> int:
    for key in _ROLE_MEMORY_KEYS.get(role, ()):
        value = _positive_int(level.get(key))
        if value:
            return value
    return fallback



def _chunks(total: int, tile: int) -> Tuple[int, ...]:
    total = max(1, int(total))
    tile = max(1, min(int(tile), total))
    full, rem = divmod(total, tile)
    result: List[int] = [tile] * full
    if rem:
        result.append(rem)
    return tuple(result)



def _bytes_for_values(values: int, bits: int) -> int:
    return int(ceil(max(0, values) * max(1, bits) / 8.0))



def _scaled_bias_bytes(spec: NodeExecutionSpec, output_channels: int, full_output_channels: int) -> int:
    region = spec.l1_regions.get("bias")
    if region is None or region.size_bytes <= 0 or full_output_channels <= 0:
        return 0
    return int(ceil(region.size_bytes * output_channels / float(full_output_channels)))



def _aggregate(groups: Iterable[TileGroup]) -> Tuple[TileGroup, ...]:
    combined: Dict[Tuple[Any, ...], TileGroup] = {}
    for group in groups:
        kernel = group.kernel
        shape_key = tuple(sorted((str(k), int(v)) for k, v in group.shape.items()))
        memory_key = tuple(sorted((str(k), int(v)) for k, v in group.role_bytes.items()))
        key = (type(kernel).__name__, repr(kernel), shape_key, memory_key)
        previous = combined.get(key)
        if previous is None:
            combined[key] = group
        else:
            combined[key] = replace(
                previous,
                multiplicity=previous.multiplicity + group.multiplicity,
            )
    return tuple(combined.values())



def _conv_groups(
    node: Any,
    spec: NodeExecutionSpec,
    kernel: KernelComputeSpec,
    level: Mapping[str, Any],
) -> Tuple[Tuple[TileGroup, ...], int, List[str]]:
    warnings: List[str] = []
    out_h_tile, out_w_tile = _spatial(
        level, "output", kernel.output_height, kernel.output_width
    )
    in_h_hint, in_w_hint = _spatial(
        level, "input", kernel.input_height, kernel.input_width
    )
    activation_out_c = _channel(level, "output", kernel.output_channels)
    activation_in_c = _channel(level, "input", kernel.input_channels)
    weight_out_c, weight_in_c = _weight_channels(
        level,
        activation_out_c,
        activation_in_c,
    )
    out_c_tile = min(kernel.output_channels, weight_out_c)
    in_c_tile = min(kernel.input_channels, weight_in_c)

    if out_c_tile != activation_out_c or in_c_tile != activation_in_c:
        warnings.append(
            "Convolution activation-channel and weight-channel tile sizes differ; "
            "DORY weights_dimensions=[C_out, C_in] is used for kernel tiling."
        )

    if kernel.is_grouped and not kernel.is_depthwise:
        # Arbitrary grouped-channel tiling requires exact group-to-channel mapping.
        # Spatial and output-channel tiling are still modeled; reduction-channel
        # tiling is conservatively left at the complete group width.
        in_c_tile = kernel.input_channels
        warnings.append(
            "Grouped-convolution input-channel tiling is not reconstructed; "
            "the complete grouped reduction is used for each spatial tile."
        )

    out_h_chunks = _chunks(kernel.output_height, out_h_tile)
    out_w_chunks = _chunks(kernel.output_width, out_w_tile)

    if kernel.is_depthwise:
        out_c_chunks = _chunks(kernel.output_channels, out_c_tile)
        in_c_chunks: Tuple[int, ...] = ()
    else:
        out_c_chunks = _chunks(kernel.output_channels, out_c_tile)
        in_c_chunks = _chunks(kernel.input_channels, in_c_tile)

    groups: List[TileGroup] = []
    full_out_c = kernel.output_channels

    for h_index, out_h in enumerate(out_h_chunks):
        top = kernel.padding_top if h_index == 0 else 0
        bottom = kernel.padding_bottom if h_index == len(out_h_chunks) - 1 else 0
        required_in_h = max(
            1,
            (out_h - 1) * kernel.stride_height
            + kernel.kernel_height
            - top
            - bottom,
        )
        # The L1 metadata describes the maximum tile. Preserve that upper bound
        # while allowing a smaller final edge tile.
        in_h = min(max(1, in_h_hint), required_in_h)

        for w_index, out_w in enumerate(out_w_chunks):
            left = kernel.padding_left if w_index == 0 else 0
            right = kernel.padding_right if w_index == len(out_w_chunks) - 1 else 0
            required_in_w = max(
                1,
                (out_w - 1) * kernel.stride_width
                + kernel.kernel_width
                - left
                - right,
            )
            in_w = min(max(1, in_w_hint), required_in_w)

            if kernel.is_depthwise:
                depth_multiplier = max(1, kernel.output_channels // kernel.input_channels)
                for out_c in out_c_chunks:
                    in_c = max(1, int(ceil(out_c / float(depth_multiplier))))
                    tile_kernel = replace(
                        kernel,
                        input_height=in_h,
                        input_width=in_w,
                        input_channels=in_c,
                        output_height=out_h,
                        output_width=out_w,
                        output_channels=out_c,
                        padding_top=top,
                        padding_bottom=bottom,
                        padding_left=left,
                        padding_right=right,
                        groups=in_c,
                    )
                    role_bytes = {
                        "input": _bytes_for_values(in_h * in_w * in_c, kernel.input_bits),
                        "output": _bytes_for_values(out_h * out_w * out_c, kernel.output_bits),
                        "weights": _bytes_for_values(
                            out_c * kernel.kernel_height * kernel.kernel_width,
                            kernel.weight_bits,
                        ),
                    }
                    bias = _scaled_bias_bytes(spec, out_c, full_out_c)
                    if bias:
                        role_bytes["bias"] = bias
                    groups.append(
                        TileGroup(
                            kernel=tile_kernel,
                            multiplicity=1,
                            role_bytes=role_bytes,
                            shape={
                                "input_height": in_h,
                                "input_width": in_w,
                                "input_channels": in_c,
                                "output_height": out_h,
                                "output_width": out_w,
                                "output_channels": out_c,
                            },
                        )
                    )
            else:
                for out_c in out_c_chunks:
                    for in_c in in_c_chunks:
                        tile_groups = kernel.groups
                        if tile_groups > 1:
                            tile_groups = min(tile_groups, in_c, out_c)
                            while tile_groups > 1 and (
                                in_c % tile_groups != 0 or out_c % tile_groups != 0
                            ):
                                tile_groups -= 1
                        tile_kernel = replace(
                            kernel,
                            input_height=in_h,
                            input_width=in_w,
                            input_channels=in_c,
                            output_height=out_h,
                            output_width=out_w,
                            output_channels=out_c,
                            padding_top=top,
                            padding_bottom=bottom,
                            padding_left=left,
                            padding_right=right,
                            groups=max(1, tile_groups),
                        )
                        reduction_channels = in_c // max(1, tile_groups)
                        role_bytes = {
                            "input": _bytes_for_values(in_h * in_w * in_c, kernel.input_bits),
                            "output": _bytes_for_values(out_h * out_w * out_c, kernel.output_bits),
                            "weights": _bytes_for_values(
                                out_c
                                * reduction_channels
                                * kernel.kernel_height
                                * kernel.kernel_width,
                                kernel.weight_bits,
                            ),
                        }
                        bias = _scaled_bias_bytes(spec, out_c, full_out_c)
                        if bias:
                            role_bytes["bias"] = bias
                        groups.append(
                            TileGroup(
                                kernel=tile_kernel,
                                multiplicity=1,
                                role_bytes=role_bytes,
                                shape={
                                    "input_height": in_h,
                                    "input_width": in_w,
                                    "input_channels": in_c,
                                    "output_height": out_h,
                                    "output_width": out_w,
                                    "output_channels": out_c,
                                },
                            )
                        )

    return _aggregate(groups), len(groups), warnings



def _pool_groups(
    spec: NodeExecutionSpec,
    kernel: PoolKernelSpec,
    level: Mapping[str, Any],
) -> Tuple[Tuple[TileGroup, ...], int, List[str]]:
    out_h_tile, out_w_tile = _spatial(
        level, "output", kernel.output_height, kernel.output_width
    )
    in_h_hint, in_w_hint = _spatial(
        level, "input", kernel.input_height, kernel.input_width
    )
    channel_tile = min(kernel.channels, _channel(level, "output", kernel.channels))

    groups: List[TileGroup] = []
    out_h_chunks = _chunks(kernel.output_height, out_h_tile)
    out_w_chunks = _chunks(kernel.output_width, out_w_tile)
    channel_chunks = _chunks(kernel.channels, channel_tile)

    for h_index, out_h in enumerate(out_h_chunks):
        top = kernel.padding_top if h_index == 0 else 0
        bottom = kernel.padding_bottom if h_index == len(out_h_chunks) - 1 else 0
        required_in_h = max(
            1,
            (out_h - 1) * kernel.stride_height
            + kernel.kernel_height
            - top
            - bottom,
        )
        in_h = min(max(1, in_h_hint), required_in_h)
        for w_index, out_w in enumerate(out_w_chunks):
            left = kernel.padding_left if w_index == 0 else 0
            right = kernel.padding_right if w_index == len(out_w_chunks) - 1 else 0
            required_in_w = max(
                1,
                (out_w - 1) * kernel.stride_width
                + kernel.kernel_width
                - left
                - right,
            )
            in_w = min(max(1, in_w_hint), required_in_w)
            for channels in channel_chunks:
                tile_kernel = replace(
                    kernel,
                    input_height=in_h,
                    input_width=in_w,
                    channels=channels,
                    output_height=out_h,
                    output_width=out_w,
                    padding_top=top,
                    padding_bottom=bottom,
                    padding_left=left,
                    padding_right=right,
                )
                groups.append(
                    TileGroup(
                        kernel=tile_kernel,
                        multiplicity=1,
                        role_bytes={
                            "input": _bytes_for_values(
                                in_h * in_w * channels, kernel.input_bits
                            ),
                            "output": _bytes_for_values(
                                out_h * out_w * channels, kernel.output_bits
                            ),
                        },
                        shape={
                            "input_height": in_h,
                            "input_width": in_w,
                            "channels": channels,
                            "output_height": out_h,
                            "output_width": out_w,
                        },
                    )
                )
    return _aggregate(groups), len(groups), []



def _add_groups(
    kernel: AddKernelSpec,
    level: Mapping[str, Any],
) -> Tuple[Tuple[TileGroup, ...], int, List[str]]:
    tile_h, tile_w = _spatial(level, "output", kernel.height, kernel.width)
    channels = min(kernel.channels, _channel(level, "output", kernel.channels))
    groups: List[TileGroup] = []
    for height in _chunks(kernel.height, tile_h):
        for width in _chunks(kernel.width, tile_w):
            for channel_count in _chunks(kernel.channels, channels):
                tile_kernel = replace(
                    kernel,
                    height=height,
                    width=width,
                    channels=channel_count,
                )
                input1 = _bytes_for_values(
                    height * width * channel_count, kernel.input1_bits
                )
                input2 = _bytes_for_values(
                    height * width * channel_count, kernel.input2_bits
                )
                output = _bytes_for_values(
                    height * width * channel_count, kernel.output_bits
                )
                groups.append(
                    TileGroup(
                        kernel=tile_kernel,
                        multiplicity=1,
                        role_bytes={
                            "input": input1,
                            "second_input": input2,
                            "output": output,
                        },
                        shape={
                            "height": height,
                            "width": width,
                            "channels": channel_count,
                        },
                    )
                )
    return _aggregate(groups), len(groups), []



def _linear_groups(
    spec: NodeExecutionSpec,
    kernel: LinearKernelSpec,
    level: Mapping[str, Any],
) -> Tuple[Tuple[TileGroup, ...], int, List[str]]:
    warnings: List[str] = []

    # DORY stores FC weight tiles as [output_neurons, input_features].
    output_tile, input_tile = _weight_channels(
        level, kernel.output_neurons, kernel.input_features
    )

    # output_dimensions is [C, H, W]; for an FC node C is the neuron count.
    output_tile = min(
        output_tile,
        _channel(level, "output", kernel.output_neurons),
    )

    weight_tile_bytes = _memory(level, "weights", 0)
    bytes_per_neuron = _bytes_for_values(kernel.input_features, kernel.weight_bits)
    if weight_tile_bytes > 0 and bytes_per_neuron > 0:
        inferred = max(1, weight_tile_bytes // bytes_per_neuron)
        output_tile = min(output_tile, inferred)

    full_input_bytes = _bytes_for_values(kernel.input_features, kernel.input_bits)
    if input_tile < kernel.input_features:
        warnings.append(
            "The fully-connected reduction dimension is tiled according to "
            "weights_dimensions[1]. The supplied PULP-NN linear kernel has no "
            "partial-accumulation interface, so tile-shape compute must not be "
            "summed without parsing the generated wrapper."
        )

    groups: List[TileGroup] = []
    for neurons in _chunks(kernel.output_neurons, max(1, output_tile)):
        tile_kernel = replace(kernel, output_neurons=neurons)
        role_bytes = {
            "input": full_input_bytes,
            "weights": _bytes_for_values(
                kernel.input_features * neurons, kernel.weight_bits
            ),
            "output": _bytes_for_values(neurons, kernel.output_bits),
        }
        bias = _scaled_bias_bytes(spec, neurons, kernel.output_neurons)
        if bias:
            role_bytes["bias"] = bias
        groups.append(
            TileGroup(
                kernel=tile_kernel,
                multiplicity=1,
                role_bytes=role_bytes,
                shape={
                    "input_features": kernel.input_features,
                    "output_neurons": neurons,
                },
            )
        )
    return _aggregate(groups), len(groups), warnings


def build_tile_plan(
    node: Any,
    spec: NodeExecutionSpec,
    automatic_metadata: Mapping[str, Any],
    config: TilingModelConfig,
) -> TilePlan:
    level_for_count = _level(node, config.level_name)
    count_candidates = [
        _positive_int(automatic_metadata.get("total_tiles")),
        _positive_int(spec.total_tiles),
        _positive_int(getattr(node, "total_tiles", 0)),
        _positive_int(getattr(node, "n_tiles", 0)),
    ]
    parsed_total = max([1] + count_candidates)
    parent_name = _next_level_name(config.level_name)

    if not config.enabled or spec.compute_kernel is None:
        return TilePlan(
            enabled=False,
            source="disabled",
            parsed_total_tiles=parsed_total,
            derived_total_tiles=1,
            runtime_total_tiles=1,
            groups=(),
            level_name=config.level_name,
            parent_level_name=parent_name,
        )

    level = _level(node, config.level_name)
    if not level:
        if parsed_total <= 1:
            return TilePlan(
                enabled=False,
                source="single_tile",
                parsed_total_tiles=parsed_total,
                derived_total_tiles=1,
                runtime_total_tiles=1,
                groups=(),
                level_name=config.level_name,
                parent_level_name=parent_name,
            )
        return TilePlan(
            enabled=True,
            source="source_count_only",
            parsed_total_tiles=parsed_total,
            derived_total_tiles=1,
            runtime_total_tiles=parsed_total,
            groups=(),
            warnings=(
                "Multiple tiles were reported, but DORY tile dimensions were "
                "not available; dimensional reuse cannot be reconstructed.",
            ),
            level_name=config.level_name,
            parent_level_name=parent_name,
        )

    kernel = spec.compute_kernel
    loop_counts, inner_loops, outer_loops, parent_name, loop_warnings = (
        derive_loop_counts(node, kernel, config.level_name)
    )

    if isinstance(kernel, KernelComputeSpec):
        groups, group_total, warnings = _conv_groups(node, spec, kernel, level)
    elif isinstance(kernel, PoolKernelSpec):
        groups, group_total, warnings = _pool_groups(spec, kernel, level)
    elif isinstance(kernel, AddKernelSpec):
        groups, group_total, warnings = _add_groups(kernel, level)
    elif isinstance(kernel, LinearKernelSpec):
        groups, group_total, warnings = _linear_groups(spec, kernel, level)
    else:
        groups, group_total, warnings = (), 1, []

    warnings.extend(loop_warnings)
    derived_total = max(1, loop_counts.kernel_calls)

    # Exact HW_node dimensions are more informative than one scalar source
    # count. Use the dimensional count whenever it is available. The parsed
    # count remains a diagnostic and a fallback when dimensions are absent.
    if derived_total > 1:
        runtime_total = derived_total
        source = "dory_dimensional_loops"
    elif parsed_total > 1:
        runtime_total = parsed_total
        source = "generated_tile_count"
    else:
        runtime_total = 1
        source = "single_tile"

    if parsed_total > 1 and derived_total > 1 and parsed_total != derived_total:
        warnings.append(
            "Generated source reports %d tiles, while exact DORY CHW/weight "
            "dimensions imply %d kernel calls. The dimensional loop count is "
            "used; inspect the generated wrapper if the discrepancy persists."
            % (parsed_total, derived_total)
        )

    if group_total > 1 and group_total != derived_total:
        warnings.append(
            "Explicit tile-shape enumeration contains %d calls, while the "
            "dimensional loop model contains %d. Loop counts are authoritative "
            "for DMA/control; shape groups are used only for average tile sizes."
            % (group_total, derived_total)
        )

    if runtime_total <= 1:
        return TilePlan(
            enabled=False,
            source="single_tile",
            parsed_total_tiles=parsed_total,
            derived_total_tiles=derived_total,
            runtime_total_tiles=1,
            groups=(),
            warnings=tuple(warnings),
            loop_counts=loop_counts,
            inner_loop_counts=inner_loops,
            outer_loop_counts=outer_loops,
            level_name=config.level_name,
            parent_level_name=parent_name,
        )

    return TilePlan(
        enabled=True,
        source=source,
        parsed_total_tiles=parsed_total,
        derived_total_tiles=derived_total,
        runtime_total_tiles=runtime_total,
        groups=groups,
        warnings=tuple(warnings),
        loop_counts=loop_counts,
        inner_loop_counts=inner_loops,
        outer_loop_counts=outer_loops,
        level_name=config.level_name,
        parent_level_name=parent_name,
    )


def tile_regions(
    base_regions: Mapping[str, L1RegionSpec],
    group: TileGroup,
) -> Dict[str, L1RegionSpec]:
    """Resize source-derived L1 regions to one tile while preserving offsets."""

    regions: Dict[str, L1RegionSpec] = {}
    for name, region in base_regions.items():
        size = group.role_bytes.get(name, region.size_bytes)
        if name == "im2col" and isinstance(group.kernel, KernelComputeSpec):
            size = group.kernel.im2col_bytes_per_core
        if name == "wt_buffer" and isinstance(group.kernel, KernelComputeSpec):
            size = 2 * group.kernel.kernel_height * group.kernel.kernel_width
        regions[name] = replace(region, size_bytes=max(0, int(size)))
    return regions



def transfer_repetitions(plan: TilePlan, role: str) -> int:
    """Return a role-specific transfer count from the dimensional loop nest."""

    if not plan.enabled:
        return 1
    if plan.derived_total_tiles > 1:
        return plan.loop_counts.repetitions_for_role(role)
    return max(1, plan.runtime_total_tiles)



def total_role_bytes(plan: TilePlan) -> Dict[str, int]:
    """Estimate physical bytes while respecting tensor reuse.

    ``TileGroup`` enumerates kernel calls, so summing its role bytes directly
    repeats input data across output-channel tiles, weights across spatial
    tiles, and output data across reduction-channel tiles.  We instead compute
    an average tile payload and multiply it by the role-specific repetition
    count derived from ``TileLoopCounts``.
    """

    raw_totals: Dict[str, int] = {}
    raw_calls = 0
    for group in plan.groups:
        multiplicity = max(1, int(group.multiplicity))
        raw_calls += multiplicity
        for role, size in group.role_bytes.items():
            raw_totals[role] = (
                raw_totals.get(role, 0) + int(size) * multiplicity
            )

    if not raw_totals:
        return {}

    denominator = max(1, raw_calls)
    totals: Dict[str, int] = {}
    for role, raw_size in raw_totals.items():
        average_tile_bytes = raw_size / float(denominator)
        repetitions = transfer_repetitions(plan, role)
        totals[role] = int(ceil(average_tile_bytes * repetitions))
    return totals


def level_memory_footprint(node: Any, level_name: str) -> int:
    """Return the sum of tensor allocations recorded for one DORY level."""

    level = _level(node, level_name)
    if not level:
        return 0
    keys = (
        "input_activation_memory",
        "second_input_activation_memory",
        "output_activation_memory",
        "weight_memory",
        "bias_memory",
        "constants_memory",
        "lut_memory",
    )
    total = 0
    seen = False
    for key in keys:
        value = _positive_int(level.get(key))
        if value:
            total += value
            seen = True
    if seen:
        return total
    # Fallback for alternate DORY dictionaries.
    for key, raw in level.items():
        if str(key).endswith("_memory"):
            total += _positive_int(raw)
    return total



def level_is_tiled(
    node: Any,
    kernel: Union[ComputeKernelSpec, None],
    level_name: str,
) -> bool:
    """Check whether a memory level stores a smaller execution tile."""

    if kernel is None:
        return False
    level = _level(node, level_name)
    if not level:
        return False

    if isinstance(kernel, KernelComputeSpec):
        out_h, out_w = _spatial(
            level, "output", kernel.output_height, kernel.output_width
        )
        out_c = _channel(level, "output", kernel.output_channels)
        in_c = _channel(level, "input", kernel.input_channels)
        return (
            out_h < kernel.output_height
            or out_w < kernel.output_width
            or out_c < kernel.output_channels
            or in_c < kernel.input_channels
        )
    if isinstance(kernel, PoolKernelSpec):
        out_h, out_w = _spatial(
            level, "output", kernel.output_height, kernel.output_width
        )
        channels = _channel(level, "output", kernel.channels)
        return (
            out_h < kernel.output_height
            or out_w < kernel.output_width
            or channels < kernel.channels
        )
    if isinstance(kernel, AddKernelSpec):
        out_h, out_w = _spatial(level, "output", kernel.height, kernel.width)
        channels = _channel(level, "output", kernel.channels)
        return out_h < kernel.height or out_w < kernel.width or channels < kernel.channels
    if isinstance(kernel, LinearKernelSpec):
        output_dims = _dims(level, "output")
        if output_dims:
            product = 1
            for value in output_dims:
                product *= value
            return product < kernel.output_neurons
        weight_tile = _memory(level, "weights", 0)
        full_weight = _bytes_for_values(
            kernel.input_features * kernel.output_neurons,
            kernel.weight_bits,
        )
        return bool(weight_tile and weight_tile < full_weight)
    return False
