

import numpy as np
from typing import Any
from dataclasses import dataclass
from ortools.constraint_solver import pywrapcp
from dory.Hardware_targets.PULP.Common.Tiler.tiler_common import PULPTilerCommon




@dataclass(frozen=True)
class AddTileShape:
    """Concrete tile selected for an element-wise addition."""

    channels: int
    height: int
    width: int


@dataclass(frozen=True)
class AddMemoryBreakdown:
    """Concrete L1 memory footprint for one addition tile, in bytes."""

    input_1: int = 0
    input_2: int = 0
    output: int = 0
    constants: int = 0
    fixed_overhead: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_1
            + self.input_2
            + self.output
            + self.constants
            + self.fixed_overhead
        )


@dataclass(frozen=True)
class SymbolicAddMemory:
    """OR-Tools expressions for the memory footprint of an addition tile."""

    input_1: Any
    input_2: Any
    output: Any
    constants: Any
    fixed_overhead: Any
    total: Any
    

class Tiler_Add_PULP(PULPTilerCommon):
    """Generate L3->L2 or L2->L1 tiling for element-wise addition."""
    
    def __init__(self,tiler):
        self.__dict__ = tiler.__dict__


    def get_tiling(self, level: int):
        if level == 3:
            return self.get_tiling_add_l3()
        if level == 2:
            return self.get_tiling_add_l2()
        raise ValueError(
            "Invalid tiling level: expected 3 for L3->L2 or 2 for L2->L1."
        )


    def get_tiling_add_l3(
        self,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return the full layer if it fits; spatial L3 tiling is unsupported."""

        l2_memory = self._available_l2_memory()
        input_from_l3 = self._input_from_l3()

        if self._full_layer_l2_bytes() <= l2_memory and not input_from_l3:
            return self._full_layer_tile()

        raise ValueError(
            "Add: the complete layer does not fit in L2, and L3->L2 tiling "
            "for addition is not implemented."
        )

    def _input_from_l3(self) -> bool:
        """Infer whether the previous output is tiled between L3 and L2."""

        previous_l3_output = self.previous_HW_node.tiling_dimensions["L3"][
            "output_dimensions"
        ]
        previous_l2_output = self.previous_HW_node.tiling_dimensions["L2"][
            "output_dimensions"
        ]

        comes_from_l3 = (
            previous_l2_output is not None
            and previous_l3_output != previous_l2_output
        )

        if comes_from_l3:
            self.HW_node.L3_input = 1

        return comes_from_l3

    def _full_layer_l2_bytes(self) -> int:
        return (
            self.HW_node.input_activation_memory
            + self.HW_node.output_activation_memory
            + self.HW_node.constants_memory
        )

    def _full_layer_tile(
        self,
    ) -> tuple[list[int], list[int], list[int]]:
        return (
            [],
            [
                self.HW_node.input_channels,
                self.HW_node.input_dimensions[0],
                self.HW_node.input_dimensions[1],
            ],
            [
                self.HW_node.output_channels,
                self.HW_node.output_dimensions[0],
                self.HW_node.output_dimensions[1],
            ],
        )
        
        
    def get_tiling_add_l2(
        self,
    ) -> tuple[list[int], list[int], list[int]]:
        """Find an L2->L1 tile for element-wise addition."""

        l1_memory = self._available_l1_memory()

        input_dimensions = self.HW_node.tiling_dimensions["L2"][
            "input_dimensions"
        ]
        output_dimensions = self.HW_node.tiling_dimensions["L2"][
            "output_dimensions"
        ]

        in_ch, input_h, input_w = input_dimensions
        out_ch, output_h, output_w = output_dimensions

        self._validate_add_geometry(
            in_ch=in_ch,
            out_ch=out_ch,
            input_h=input_h,
            input_w=input_w,
            output_h=output_h,
            output_w=output_w,
        )

        full_tile = AddTileShape(
            channels=in_ch,
            height=input_h,
            width=input_w,
        )
        full_memory = self._concrete_l1_memory(
            tile=full_tile,
            double_buffering=1,
        )

        if full_memory.total <= l1_memory:
            return (
                [],
                list(input_dimensions),
                list(output_dimensions),
            )

        double_buffering = self.double_buffering

        solver = pywrapcp.Solver(
            "Add_L2_L1",
            pywrapcp.Solver.DefaultSolverParameters(),
        )

        variables = self._create_l2_l1_variables(
            solver=solver,
            channels=in_ch,
            height=input_h,
            width=input_w,
        )

        self._add_addition_geometry_constraints(
            solver=solver,
            variables=variables,
            channels=in_ch,
        )

        symbolic_memory = self._symbolic_l1_memory(
            tile_channels=variables["channels"],
            tile_height=variables["height"],
            tile_width=variables["width"],
            double_buffering=double_buffering,
        )
        solver.Add(symbolic_memory.total <= l1_memory)

        objective_expr = self._build_l2_l1_objective(
            solver=solver,
            variables=variables,
            memory_total=symbolic_memory.total,
        )

        tile = self._solve_l2_l1_problem(
            solver=solver,
            variables=variables,
            objective_expr=objective_expr,
        )

        if tile is None:
            raise ValueError(
                "Add: no L2->L1 tile fits in {} bytes of available L1.".format(
                    l1_memory
                )
            )

        concrete_memory = self._concrete_l1_memory(
            tile=tile,
            double_buffering=double_buffering,
        )
        if concrete_memory.total > l1_memory:
            raise RuntimeError(
                "Add solver returned an invalid tile: {} bytes required, "
                "{} bytes available. Breakdown: {}".format(
                    concrete_memory.total,
                    l1_memory,
                    concrete_memory,
                )
            )

        return (
            [],
            [tile.channels, tile.height, tile.width],
            [tile.channels, tile.height, tile.width],
        )

    @staticmethod
    def _validate_add_geometry(
        *,
        in_ch: int,
        out_ch: int,
        input_h: int,
        input_w: int,
        output_h: int,
        output_w: int,
    ) -> None:
        if in_ch != out_ch:
            raise ValueError(
                "Add requires equal input/output channels, got {} and {}.".format(
                    in_ch, out_ch
                )
            )

        if (input_h, input_w) != (output_h, output_w):
            raise ValueError(
                "Add requires equal input/output spatial dimensions, got "
                "{}x{} and {}x{}.".format(
                    input_h,
                    input_w,
                    output_h,
                    output_w,
                )
            )

    @staticmethod
    def _create_l2_l1_variables(
        *,
        solver: pywrapcp.Solver,
        channels: int,
        height: int,
        width: int,
    ) -> dict[str, Any]:
        # h_out/w_out are not independent for element-wise addition.
        return {
            "channels": solver.IntVar(1, channels, "tile_channels"),
            "height": solver.IntVar(1, height, "tile_height"),
            "width": solver.IntVar(1, width, "tile_width"),
        }

    @staticmethod
    def _add_addition_geometry_constraints(
        *,
        solver: pywrapcp.Solver,
        variables: dict[str, Any],
        channels: int,
    ) -> None:
        # Current backend tiles only spatial dimensions.
        solver.Add(variables["channels"] == channels)

    def _symbolic_l1_memory(
        self,
        *,
        tile_channels: Any,
        tile_height: Any,
        tile_width: Any,
        double_buffering: int,
    ) -> SymbolicAddMemory:
        """Build symbolic memory expressions for both inputs and the output."""

        elements = tile_channels * tile_height * tile_width

        input_1_bits = elements * self.HW_node.input_activation_bits
        input_2_bits = elements * self.HW_node.second_input_activation_bits
        output_bits = elements * self.HW_node.output_activation_bits

        input_1_bytes = double_buffering * self._symbolic_bytes_for_bits(
            input_1_bits
        )
        input_2_bytes = double_buffering * self._symbolic_bytes_for_bits(
            input_2_bits
        )
        output_bytes = double_buffering * self._symbolic_bytes_for_bits(
            output_bits
        )

        constants_bytes = self.HW_node.tiling_dimensions["L2"][
            "constants_memory"
        ]

        total = (
            input_1_bytes
            + input_2_bytes
            + output_bytes
            + constants_bytes
        )

        return SymbolicAddMemory(
            input_1=input_1_bytes,
            input_2=input_2_bytes,
            output=output_bytes,
            constants=constants_bytes,
            fixed_overhead=0,
            total=total,
        )

    def _concrete_l1_memory(
        self,
        *,
        tile: AddTileShape,
        double_buffering: int,
    ) -> AddMemoryBreakdown:
        """Evaluate memory for one known tile using Python integers."""

        elements = tile.channels * tile.height * tile.width

        input_1_bytes = double_buffering * self._bytes_for_bits(
            elements * self.HW_node.input_activation_bits
        )
        input_2_bytes = double_buffering * self._bytes_for_bits(
            elements * self.HW_node.second_input_activation_bits
        )
        output_bytes = double_buffering * self._bytes_for_bits(
            elements * self.HW_node.output_activation_bits
        )

        constants_bytes = self.HW_node.tiling_dimensions["L2"][
            "constants_memory"
        ]

        return AddMemoryBreakdown(
            input_1=input_1_bytes,
            input_2=input_2_bytes,
            output=output_bytes,
            constants=constants_bytes,
            fixed_overhead=0,
        )

    @staticmethod
    def _build_l2_l1_objective(
        *,
        solver: pywrapcp.Solver,
        variables: dict[str, Any],
        memory_total: Any,
    ) -> Any:
        """Prefer high L1 utilization, then wider and taller tiles."""

        score = (
            1_000_000 * memory_total
            + 10 * variables["width"]
            + variables["height"]
        )

        objective_expr = solver.IntVar(
            0,
            1_000_000_000_000_000,
            "objective",
        )
        solver.Add(objective_expr == score)
        return objective_expr

    @staticmethod
    def _solve_l2_l1_problem(
        *,
        solver: pywrapcp.Solver,
        variables: dict[str, Any],
        objective_expr: Any,
    ) -> AddTileShape | None:
        """Run CP search and extract one concrete tile."""

        ordered_variables = [
            variables["channels"],
            variables["height"],
            variables["width"],
        ]

        decision_builder = solver.Phase(
            ordered_variables,
            solver.CHOOSE_FIRST_UNBOUND,
            solver.ASSIGN_MIN_VALUE,
        )

        objective = solver.Maximize(objective_expr, 1)

        collector = solver.LastSolutionCollector()
        for variable in ordered_variables:
            collector.Add(variable)
        collector.AddObjective(objective_expr)

        solver.Solve(decision_builder, [objective, collector])

        if collector.SolutionCount() == 0:
            return None

        best = collector.SolutionCount() - 1

        return AddTileShape(
            channels=collector.Value(best, variables["channels"]),
            height=collector.Value(best, variables["height"]),
            width=collector.Value(best, variables["width"]),
        )
