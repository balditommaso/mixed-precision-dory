from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .calibration import get_pessimistic_cycles

def format_summary(result: Mapping[str, Any]) -> str:
    """Return only the final pessimistic latency for concise notebook output."""
    return f"{result['name']}: {int(result['pessimistic_cycles']):,} cycles"


def format_diagnostics(result: Mapping[str, Any]) -> str:
    """Return the verbose diagnostic report when explicitly requested."""
    measured = result['diagnostics']['measured_cycles']
    lines = [
        f"Node: {result['name']}",
        f"Model: {result['compute'].get('model', 'unknown')}",
        f"Cores: {result['num_cores']}",
        f"Lower bound: {result['lower_bound_cycles']:,} cycles",
        f"Expected: {result['expected_cycles']:,} cycles",
        f"Pessimistic: {result['pessimistic_cycles']:,} cycles",
    ]
    if measured is not None:
        lines.append(f"Measured: {measured:,} cycles")
        lines.append(
            f"Pessimistic/measured: "
            f"{result['diagnostics']['pessimistic_ratio_to_measured']:.3f}"
        )
    if result['warnings']:
        lines.append('Warnings:')
        lines.extend(f'  - {warning}' for warning in result['warnings'])
    return '\n'.join(lines)


def format_pessimistic(result: Mapping[str, Any]) -> str:
    """Return only the final pessimistic latency for one node."""

    return "%s: %d cycles" % (
        result["name"],
        get_pessimistic_cycles(result),
    )


def print_pessimistic_latencies(
    results: Iterable[Mapping[str, Any]],
    *,
    include_total: bool = False,
    frequency_hz: Optional[float] = None,
) -> None:
    """Print one compact latency line per node and optionally the total."""

    result_list = list(results)
    for result in result_list:
        print(format_pessimistic(result))

    if include_total:
        total_cycles = sum(get_pessimistic_cycles(result) for result in result_list)
        if frequency_hz is None:
            print("Total: %d cycles" % total_cycles)
        else:
            if frequency_hz <= 0:
                raise ValueError("frequency_hz must be positive")
            total_ms = total_cycles / float(frequency_hz) * 1e3
            print("Total: %d cycles (%.3f ms)" % (total_cycles, total_ms))

