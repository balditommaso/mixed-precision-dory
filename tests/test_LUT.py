
import json
import os
import re
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_DIR = PROJECT_ROOT / "application"
MODEL_DIR = PROJECT_ROOT / "models"

RUN_TIMEOUT_SECONDS = 360


@dataclass(frozen=True)
class NetworkCase:
    """Everything required to generate and run one DORY network."""

    name: str
    model_path: Path
    frontend: str = "QONNX"
    hardware: str = "PULP.PULP_gvsoc"
    isa: str = "mixed-sw"
    verbose: str = "None"
    platform_index: int = 1
    n_inputs: int = 1
    delta: int = 16


PLATFORMS = (
    {
        "num_cores": 8,
        "L1_capacity": 64_000,
        "L2_capacity": 512_000,
    },
    {
        "num_cores": 4,
        "L1_capacity": 64_000,
        "L2_capacity": 320_000,
    },
    {
        "num_cores": 2,
        "L1_capacity": 64_000,
        "L2_capacity": 256_000,
    },
)


NETWORK_PAIRS = (
    (
        NetworkCase(
            name="im2col",
            model_path=MODEL_DIR / "DummyConv_im2col.onnx",
        ),
        NetworkCase(
            name="lut",
            model_path=MODEL_DIR / "DummyConv_lut.onnx",
        ),
    ),
)


def validate_case(case: NetworkCase) -> None:
    if not case.model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {case.model_path}")

    if not 0 <= case.platform_index < len(PLATFORMS):
        raise ValueError(
            f"Invalid platform index {case.platform_index}; "
            f"expected 0..{len(PLATFORMS) - 1}"
        )


def generate_network(case: NetworkCase) -> int:
    """
    Generate C code using the same frontend -> HW parser -> C parser flow
    used by the ALADIN notebook.

    Returns the number of cluster cores selected for the build.
    """

    validate_case(case)
    dory_config = dory_config = {
        "BNRelu_bits": 32,
        "code reserved space": 150000,
        "input_bits": 8,
        "input_signed": False
    }
    platform = PLATFORMS[case.platform_index]

    num_cores = int(platform["num_cores"])
    l1_capacity = int(platform["L1_capacity"])
    l2_capacity = int(platform["L2_capacity"])

    frontend_module = import_module(
        f"dory.Frontend_frameworks.{case.frontend}.Parser"
    )
    onnx_to_dory = frontend_module.onnx_manager

    dory_graph = onnx_to_dory(
        str(case.model_path),
        dory_config,
        delta=case.delta,
    ).full_graph_parsing()

    hardware_module = import_module(
        f"dory.Hardware_targets.{case.hardware}.HW_Parser"
    )
    dory_to_hw = hardware_module.onnx_manager

    hw_graph = dory_to_hw(
        deepcopy(dory_graph),
        n_inputs=case.n_inputs,
        verify_checksum=False,
        L1_capacity=l1_capacity,
        L2_capacity=l2_capacity,
        config_file=dory_config,
        num_cores=num_cores,
    ).full_graph_parsing()

    c_parser_module = import_module(
        f"dory.Hardware_targets.{case.hardware}.C_Parser"
    )
    dory_hw_to_c = c_parser_module.C_Parser

    dory_hw_to_c(
        hw_graph,
        dory_config,
        verbose_level=case.verbose,
        perf_layer="Yes",
        precision_library=case.isa,
        app_directory=str(APPLICATION_DIR),
        model_dir=str(MODEL_DIR),
        n_inputs=case.n_inputs,
        L1_capacity=l1_capacity,
        L2_capacity=l2_capacity,
    ).full_graph_parsing()

    return num_cores


def run_generated_network(
    *,
    num_cores: int,
    timeout_seconds: int = RUN_TIMEOUT_SECONDS,
) -> str:
    """
    Build and run the currently generated application under GVSOC.

    The SDK setup is performed exactly as in the notebook.
    """

    shell_command = f"""
        source .devcontainer/pulp_sdk.sh
        export PATH=/dory_env/bin:$PATH
        make -C {APPLICATION_DIR} clean all run \
            platform=gvsoc CORE={num_cores}
    """

    try:
        completed = subprocess.run(
            ["bash", "-lc", shell_command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(
            "Generated application failed.\n"
            f"Exit code: {error.returncode}\n"
            f"STDOUT:\n{error.stdout}\n"
            f"STDERR:\n{error.stderr}"
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"Generated application timed out after {timeout_seconds}s.\n"
            f"Partial STDOUT:\n{error.stdout or ''}\n"
            f"Partial STDERR:\n{error.stderr or ''}"
        )

    # Keep stderr because GVSOC may report useful runtime diagnostics there.
    return completed.stdout + "\n" + completed.stderr


def parse_final_output(runtime_output: str) -> list[int]:
    """
    Extract the final output vector from DORY/GVSOC output.

    Supported examples:

        Final output:
        1 -2 3 4

        Final output: 1 -2 3 4

        Final output [0]: 1 -2 3 4
    """

    patterns = (
        r"Final\s+output[^\n:]*:\s*\n?\s*([+\-\d][+\-\d,\s]*)",
        r"Final\s+output[^\n]*\n\s*([+\-\d][+\-\d,\s]*)",
    )

    for pattern in patterns:
        match = re.search(pattern, runtime_output, flags=re.IGNORECASE)
        if match is None:
            continue

        values = [int(value) for value in re.findall(r"[+-]?\d+", match.group(1))]
        if values:
            return values

    tail = "\n".join(runtime_output.splitlines()[-100:])
    pytest.fail(
        "Could not find a final output vector in the generated runtime output.\n"
        f"Last 100 lines:\n{tail}"
    )


def generate_run_and_parse(case: NetworkCase) -> tuple[list[int], str]:
    print(
        f"\nGenerating {case.name}\n"
        f"  model:  {case.model_path}\n",
        flush=True,
    )

    num_cores = generate_network(case)
    runtime_output = run_generated_network(num_cores=num_cores)
    final_output = parse_final_output(runtime_output)

    print(
        f"{case.name} final output ({len(final_output)} values):\n"
        f"{final_output}",
        flush=True,
    )

    return final_output, runtime_output


@pytest.mark.parametrize(
    ("reference_case", "candidate_case"),
    NETWORK_PAIRS,
    ids=lambda case: case.name,
)
def test_network_outputs_match(
    reference_case: NetworkCase,
    candidate_case: NetworkCase,
    tmp_path: Path,
) -> None:
    """
    Generate and run both networks sequentially, then compare final outputs.

    Each generation overwrites ./application, which is safe because the first
    output is captured before generating the second application.
    """

    reference_output, reference_log = generate_run_and_parse(reference_case)
    candidate_output, candidate_log = generate_run_and_parse(candidate_case)

    # Persist complete logs to make a CI failure reproducible.
    (tmp_path / f"{reference_case.name}.log").write_text(
        reference_log,
        encoding="utf-8",
    )
    (tmp_path / f"{candidate_case.name}.log").write_text(
        candidate_log,
        encoding="utf-8",
    )

    assert len(reference_output) == len(candidate_output), (
        "Network output lengths differ:\n"
        f"  {reference_case.name}: {len(reference_output)}\n"
        f"  {candidate_case.name}: {len(candidate_output)}"
    )

    mismatches = [
        (index, expected, actual)
        for index, (expected, actual) in enumerate(
            zip(reference_output, candidate_output)
        )
        if expected != actual
    ]

    assert not mismatches, (
        f"Final outputs differ in {len(mismatches)} position(s).\n"
        f"First mismatches: {mismatches[:20]}\n"
        f"Reference ({reference_case.name}): {reference_output}\n"
        f"Candidate ({candidate_case.name}): {candidate_output}"
    )