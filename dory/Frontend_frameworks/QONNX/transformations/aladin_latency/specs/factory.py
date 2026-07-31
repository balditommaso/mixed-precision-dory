from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

from ..config import AutoSpecConfig
from ..descriptors import NodeExecutionSpec
from ..source import infer_pulp_kernel_name, load_generated_source_metadata
from ..utils import align_up, nonnegative_int
from .add import make_add_execution_spec
from .common import (
    default_execution_spec,
    infer_generic_l1_offsets,
    is_supported_add_node,
    is_supported_linear_node,
    is_supported_pool_node,
    is_supported_pulp_conv_node,
)
from .conv import (
    infer_l1_offsets,
    make_pulp_conv_kernel_spec,
    make_single_tile_dory_conv_spec,
    make_single_tile_dory_depthwise_spec,
)
from .linear import make_linear_execution_spec
from .pooling import make_pool_execution_spec


def build_execution_spec_automatically(node: Any, config: AutoSpecConfig) -> Tuple[NodeExecutionSpec, Dict[str, Any], List[str]]:
    """Build the execution model for one node using global rules only."""
    warnings: List[str] = []
    source_meta = load_generated_source_metadata(node, config)
    if is_supported_pulp_conv_node(node):
        kernel_name = source_meta.kernel_name or infer_pulp_kernel_name(node)
        if kernel_name is None:
            kernel_name = 'pulp_nn_conv_unknown'
        flag_relu = 'relu' in str(getattr(node, 'op_type', '')).lower()
        constant_names = {
            str(name).lower()
            for name in getattr(node, 'constant_names', ()) or ()
        }
        flag_batch_norm = bool({'kappa', 'lambda'} & constant_names)
        provisional_kernel = make_pulp_conv_kernel_spec(
            node,
            kernel_name,
            flag_relu=flag_relu,
            flag_batch_norm=flag_batch_norm,
        )
        offsets = dict(source_meta.l1_offsets)
        if not offsets:
            offsets = infer_l1_offsets(node, provisional_kernel, config)
            warnings.append(
                'Generated L1 offsets were unavailable; a deterministic '
                'packed layout was inferred for the bank model.'
            )

        if provisional_kernel.is_depthwise:
            spec = make_single_tile_dory_depthwise_spec(
                node,
                kernel_name,
                l1_offsets=offsets,
                flag_relu=flag_relu,
                flag_batch_norm=flag_batch_norm,
                weight_correlation=config.depthwise_weight_correlation,
                input_correlation=config.depthwise_input_correlation,
                im2col_correlation=config.depthwise_im2col_correlation,
                output_correlation=config.depthwise_output_correlation,
                bias_correlation=config.depthwise_weight_correlation,
            )
        else:
            spec = make_single_tile_dory_conv_spec(
                node,
                kernel_name,
                l1_offsets=offsets,
                flag_relu=flag_relu,
                flag_batch_norm=flag_batch_norm,
                weight_correlation=config.weight_correlation,
                input_correlation=config.input_correlation,
                im2col_correlation=config.im2col_correlation,
                output_correlation=config.output_correlation,
                bias_correlation=config.bias_correlation,
            )

        total_tiles = source_meta.total_tiles or 1
        if total_tiles > 1:
            warnings.append(
                f'Generated code reports {total_tiles} runtime tiles. Tile shapes and '
                'tile-dependent DMA/control costs are evaluated by the engine.'
            )
        external_barriers = (
            source_meta.team_barriers_outside_kernel
            if source_meta.team_barriers_outside_kernel is not None
            else config.default_team_barriers_outside_kernel
        )
        events = dict(source_meta.control_events)
        if not events:
            events = {
                'dma_allocations': config.default_dma_allocations,
                'dma_frees': config.default_dma_frees,
                'kernel_calls': config.default_kernel_calls,
            }
        spec = replace(
            spec,
            total_tiles=total_tiles,
            team_barriers_outside_kernel=external_barriers,
            team_barriers_inside_kernel=config.default_team_barriers_inside_pulp_conv,
            control_events=events,
        )
        metadata = {
            'source': 'generated_code' if source_meta.source_path else 'node_inference',
            'source_path': source_meta.source_path,
            'kernel_name': kernel_name,
            'kernel_kind': provisional_kernel.kernel_kind,
            'groups': provisional_kernel.groups,
            'l1_offsets': offsets,
            'total_tiles': total_tiles,
        }
        return spec, metadata, warnings


    if is_supported_add_node(node, source_meta.kernel_name):
        kernel_name = source_meta.kernel_name or 'pulp_nn_add_i8_i8_i8'
        offsets = dict(source_meta.l1_offsets)
        if not offsets:
            offsets = infer_generic_l1_offsets(node, config)
            if 'second_input' not in offsets:
                first_size = nonnegative_int(getattr(node, 'input_activation_memory', 0))
                offsets['second_input'] = align_up(
                    offsets.get('input', 0) + first_size + config.l1_guard_bytes,
                    config.l1_alignment_bytes,
                )
                offsets['output'] = align_up(
                    offsets['second_input'] + first_size + config.l1_guard_bytes,
                    config.l1_alignment_bytes,
                )
            warnings.append(
                'Generated L1 offsets were unavailable; a deterministic '
                'packed add layout was inferred for the bank model.'
            )
        spec = make_add_execution_spec(node, kernel_name, offsets, config)
        total_tiles = source_meta.total_tiles or 1
        events = dict(source_meta.control_events) or {
            'dma_allocations': config.default_dma_allocations,
            'dma_frees': config.default_dma_frees,
            'kernel_calls': config.default_kernel_calls,
        }
        external_barriers = (
            source_meta.team_barriers_outside_kernel
            if source_meta.team_barriers_outside_kernel is not None
            else config.default_team_barriers_outside_kernel
        )
        spec = replace(
            spec, total_tiles=total_tiles,
            team_barriers_outside_kernel=external_barriers,
            team_barriers_inside_kernel=1, control_events=events,
        )
        metadata = {
            'source': 'generated_code' if source_meta.source_path else 'node_inference',
            'source_path': source_meta.source_path,
            'kernel_name': kernel_name,
            'kernel_kind': 'quantized_add',
            'groups': 1,
            'l1_offsets': offsets,
            'total_tiles': total_tiles,
        }
        return spec, metadata, warnings

    if is_supported_pool_node(node):
        kernel_name = source_meta.kernel_name or 'pulp_nn_maxpool_i8'
        offsets = dict(source_meta.l1_offsets)
        if not offsets:
            offsets = infer_generic_l1_offsets(node, config)
            warnings.append(
                'Generated L1 offsets were unavailable; a deterministic '
                'packed pooling layout was inferred for the bank model.'
            )
        spec = make_pool_execution_spec(node, kernel_name, offsets, config)
        total_tiles = source_meta.total_tiles or 1
        events = dict(source_meta.control_events) or {
            'dma_allocations': config.default_dma_allocations,
            'dma_frees': config.default_dma_frees,
            'kernel_calls': config.default_kernel_calls,
        }
        external_barriers = (
            source_meta.team_barriers_outside_kernel
            if source_meta.team_barriers_outside_kernel is not None
            else config.default_team_barriers_outside_kernel
        )
        spec = replace(
            spec,
            total_tiles=total_tiles,
            team_barriers_outside_kernel=external_barriers,
            team_barriers_inside_kernel=2,
            control_events=events,
        )
        metadata = {
            'source': 'generated_code' if source_meta.source_path else 'node_inference',
            'source_path': source_meta.source_path,
            'kernel_name': kernel_name,
            'kernel_kind': 'maxpool',
            'groups': 1,
            'l1_offsets': offsets,
            'total_tiles': total_tiles,
        }
        return spec, metadata, warnings

    if is_supported_linear_node(node):
        kernel_name = source_meta.kernel_name or 'pulp_nn_linear_i8_i8_i8'
        offsets = dict(source_meta.l1_offsets)
        if not offsets:
            offsets = infer_generic_l1_offsets(node, config)
            warnings.append(
                'Generated L1 offsets were unavailable; a deterministic '
                'packed linear layout was inferred for the bank model.'
            )
        spec = make_linear_execution_spec(node, kernel_name, offsets, config)
        total_tiles = source_meta.total_tiles or 1
        events = dict(source_meta.control_events) or {
            'dma_allocations': config.default_dma_allocations,
            'dma_frees': config.default_dma_frees,
            'kernel_calls': config.default_kernel_calls,
        }
        external_barriers = (
            source_meta.team_barriers_outside_kernel
            if source_meta.team_barriers_outside_kernel is not None
            else config.default_team_barriers_outside_kernel
        )
        spec = replace(
            spec,
            total_tiles=total_tiles,
            team_barriers_outside_kernel=external_barriers,
            team_barriers_inside_kernel=1,
            control_events=events,
        )
        metadata = {
            'source': 'generated_code' if source_meta.source_path else 'node_inference',
            'source_path': source_meta.source_path,
            'kernel_name': kernel_name,
            'kernel_kind': 'linear',
            'groups': 1,
            'l1_offsets': offsets,
            'total_tiles': total_tiles,
        }
        return spec, metadata, warnings

    spec = default_execution_spec(node)
    metadata = {
        'source': 'generic_fallback',
        'source_path': source_meta.source_path,
        'kernel_name': None,
        'kernel_kind': 'generic',
        'l1_offsets': {},
        'total_tiles': source_meta.total_tiles or 1,
    }
    warnings.append(
        'No source-level kernel model is registered for this operation; '
        'using the operation-specific or MAC-only fallback.'
    )
    return spec, metadata, warnings
