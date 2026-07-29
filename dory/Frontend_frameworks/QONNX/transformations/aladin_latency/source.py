from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Union

from .config import AutoSpecConfig
from .descriptors import NodeSourceMetadata


def _node_source_candidates(node: Any, config: AutoSpecConfig) -> List[Path]:
    if config.generated_code_dir is None:
        return []
    root = Path(config.generated_code_dir)
    names: List[str] = []
    for value in (getattr(node, 'name', None), getattr(node, 'prefixed_name', None)):
        if value and str(value) not in names:
            names.append(str(value))
    candidates = [root / f'{name}{config.source_suffix}' for name in names]
    for candidate in candidates:
        if candidate.is_file():
            return [candidate]
    matches: List[Path] = []
    for name in names:
        matches.extend(root.rglob(f'{name}{config.source_suffix}'))
    return sorted(set(matches))


def parse_generated_layer_source(source: str, *, source_path: Union[str, None]=None) -> NodeSourceMetadata:
    """Extract static metadata without requiring a per-node specification."""
    kernel_matches = re.findall('\\b(pulp_nn_(?:conv|depthwise|linear|matmul|maxpool|avgpool|add)_[A-Za-z0-9_]+)\\s*\\(', source)
    kernel_name = kernel_matches[0] if kernel_matches else None
    total_tiles_match = re.search('\\btotal_tiles\\s*=\\s*(\\d+)\\s*;', source)
    total_tiles = int(total_tiles_match.group(1)) if total_tiles_match else None
    offsets: Dict[str, int] = {}
    pointer_variables = {
        'input': 'x',
        'output': 'y',
        'weights': 'W',
        'bias': 'b',
        'im2col': 'im2col',
        'wt_buffer': 'wt_buffer',
    }
    for role, variable in pointer_variables.items():
        match = re.search(f'\\b{re.escape(variable)}\\s*=\\s*[^;]*?\\bl1_buffer\\s*\\+\\s*\\(?\\s*(\\d+)', source)
        if match:
            offsets[role] = int(match.group(1))
    for variable in ('x2', 'x_2', 'x_second', 'x2_input'):
        match = re.search(
            f'\\b{re.escape(variable)}\\s*=\\s*[^;]*?\\bl1_buffer\\s*\\+\\s*\\(?\\s*(\\d+)',
            source,
        )
        if match:
            offsets['second_input'] = int(match.group(1))
            break
    if 'wt_buffer' not in offsets:
        wt_match = re.search(r'\b(?:pWtBuffer|wt_buffer|pWt)\s*=\s*[^;]*?\bl1_buffer\s*\+\s*\(?\s*(\d+)', source)
        if wt_match:
            offsets['wt_buffer'] = int(wt_match.group(1))
    control_events = {'dma_allocations': len(re.findall('\\bdory_dma_allocate\\s*\\(', source)), 'dma_frees': len(re.findall('\\bdory_dma_free\\s*\\(', source)), 'kernel_calls': len(kernel_matches)}
    return NodeSourceMetadata(source_path=source_path, kernel_name=kernel_name, total_tiles=total_tiles, l1_offsets=offsets, team_barriers_outside_kernel=len(re.findall('\\bpi_cl_team_barrier\\s*\\(', source)), control_events=control_events)


def load_generated_source_metadata(node: Any, config: AutoSpecConfig) -> NodeSourceMetadata:
    if not config.parse_generated_code:
        return NodeSourceMetadata()
    candidates = _node_source_candidates(node, config)
    if not candidates:
        return NodeSourceMetadata()
    path = candidates[0]
    try:
        source = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return NodeSourceMetadata()
    return parse_generated_layer_source(source, source_path=str(path))


def _type_prefix(type_name: Any, bits: int, *, default_signed: bool) -> str:
    text = str(type_name or '').lower()
    if 'uint' in text or text.startswith('u'):
        return f'u{bits}'
    if 'int' in text or text.startswith('i'):
        return f'i{bits}'
    return f"{('i' if default_signed else 'u')}{bits}"


def infer_pulp_kernel_name(node: Any) -> Union[str, None]:
    """Infer a PULP-NN kernel family from generic HW_node metadata."""
    op_type = str(getattr(node, 'op_type', '')).lower()
    if 'conv' not in op_type:
        return None
    input_bits = int(getattr(node, 'input_activation_bits', 8) or 8)
    output_bits = int(getattr(node, 'output_activation_bits', 8) or 8)
    weight_bits = int(getattr(node, 'weight_bits', 8) or 8)
    input_code = _type_prefix(getattr(node, 'input_activation_type', None), input_bits, default_signed=True)
    output_code = _type_prefix(getattr(node, 'output_activation_type', None), output_bits, default_signed=False)
    weight_code = _type_prefix(getattr(node, 'weight_type', None), weight_bits, default_signed=True)
    groups = int(getattr(node, 'group', 1) or 1)
    input_channels = int(getattr(node, 'input_channels', 1) or 1)
    output_channels = int(getattr(node, 'output_channels', 1) or 1)
    is_depthwise = (
        groups == input_channels
        and output_channels % max(1, input_channels) == 0
    )
    family = 'depthwise' if is_depthwise else 'conv'
    return f'pulp_nn_{family}_{input_code}_{output_code}_{weight_code}'
