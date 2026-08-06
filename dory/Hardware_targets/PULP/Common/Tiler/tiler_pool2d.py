from dataclasses import dataclass
from typing import Any
from dory.Hardware_targets.PULP.Common.Tiler.tiler_common import PULPTilerCommon
from ortools.constraint_solver import pywrapcp



@dataclass(frozen=True)
class PoolTileShape:
    """Concrete pooling tile dimensions after solving."""

    channels: int
    h_in: int
    w_in: int
    h_out: int
    w_out: int


@dataclass(frozen=True)
class PoolMemoryBreakdown:
    """Concrete memory footprint for one pooling tile, in bytes."""

    input: int = 0
    output: int = 0
    constants: int = 0
    fixed_overhead: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.constants + self.fixed_overhead


@dataclass(frozen=True)
class SymbolicPoolMemory:
    """Symbolic OR-Tools memory expressions for a pooling tile."""

    input: Any
    output: Any
    constants: Any
    fixed_overhead: Any
    total: Any



class Tiler_Pool2D_PULP(PULPTilerCommon):
    """Generate L3->L2 and L2->L1 pooling tiles for PULP targets."""

    def __init__(self, tiler):
        self.__dict__ = tiler.__dict__

    def get_tiling(
        self,
        level: int,
    ) -> tuple[list[int], list[int], list[int]]:
        if level == 3:
            return self.get_tiling_pool2d_l3()

        if level == 2:
            return self.get_tiling_pool2d_l2()

        raise ValueError(
            "Invalid tiling level: expected 3 for L3->L2 or 2 for L2->L1."
        )

    # ------------------------------------------------------------------
    # L3 -> L2
    # ------------------------------------------------------------------

    def get_tiling_pool2d_l3(
        self,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Find a height-only L3->L2 pooling tile.

        The inherited backend keeps:
        - all channels;
        - full tensor width;
        - optional tiling only along height.
        """

        l2_memory = self._available_l2_memory()
        input_from_l3, producer_l2_output_bytes = self._input_from_l3()

        if self._full_layer_l2_bytes() <= l2_memory and not input_from_l3:
            return self._full_layer_tile()

        ks = self.HW_node.kernel_shape
        inp_dim = self.HW_node.input_dimensions
        out_dim = self.HW_node.output_dimensions
        channels = self.HW_node.output_channels
        strides = self.HW_node.strides

        for strategy in range(2):
            solver = pywrapcp.Solver(
                "Pool2D_L3_L2",
                pywrapcp.Solver.DefaultSolverParameters(),
            )

            tile_h_out = solver.IntVar(1, out_dim[0], "tile_h_out")

            if input_from_l3:
                tile_h_in = solver.IntVar(
                    ks[0],
                    inp_dim[0],
                    "tile_h_in",
                )
                solver.Add((tile_h_in - ks[0]) % strides[0] == 0)
                db_input = 2
            else:
                tile_h_in = solver.IntVar(
                    inp_dim[0],
                    inp_dim[0],
                    "tile_h_in",
                )
                db_input = 1

            db_output = self._configure_l3_strategy(
                solver=solver,
                strategy=strategy,
                db_input=db_input,
                tile_h_out=tile_h_out,
                full_h_out=out_dim[0],
            )

            symbolic_memory = self._symbolic_l3_memory(
                tile_h_in=tile_h_in,
                tile_h_out=tile_h_out,
                db_input=db_input,
                db_output=db_output,
            )

            solver.Add(symbolic_memory.total <= l2_memory)

            if input_from_l3:
                # Preserve the inherited restriction that the buffered input
                # tile cannot exceed the producer's L2 output allocation.
                solver.Add(
                    symbolic_memory.input <= producer_l2_output_bytes
                )

            if db_input == 2 and db_output == 2:
                solver.Add(
                    tile_h_out * strides[0]
                    == tile_h_in
                    - (ks[0] - 1)
                    + (strides[0] - 1)
                )

            objective_expr = self._build_l3_objective(
                solver=solver,
                memory_total=symbolic_memory.total,
                tile_h_in=tile_h_in,
                tile_h_out=tile_h_out,
            )

            solution = self._solve_l3_problem(
                solver=solver,
                tile_h_in=tile_h_in,
                tile_h_out=tile_h_out,
                objective_expr=objective_expr,
            )

            if solution is not None:
                h_in, h_out = solution
                return (
                    [],
                    [channels, h_in, inp_dim[1]],
                    [channels, h_out, out_dim[1]],
                )

        raise ValueError(
            "Pool2D: no L3->L2 tiling strategy produced a feasible tile."
        )

    def _input_from_l3(self) -> tuple[bool, int]:
        """
        Infer whether input data comes from L3.

        Architectural limitation:
        ``previous_HW_node`` may not be the actual producer in a branched graph.
        """

        previous_l2_output = self.previous_HW_node.tiling_dimensions["L2"][
            "output_dimensions"
        ]
        previous_l3_output = self.previous_HW_node.tiling_dimensions["L3"][
            "output_dimensions"
        ]

        comes_from_l3 = (
            previous_l2_output is not None
            and previous_l2_output != previous_l3_output
        )

        if not comes_from_l3:
            return False, 0

        self.HW_node.L3_input = 1
        producer_l2_output_bytes = int(
            self.previous_HW_node.tiling_dimensions["L2"][
                "output_activation_memory"
            ]
        )
        return True, producer_l2_output_bytes

    @staticmethod
    def _configure_l3_strategy(
        *,
        solver: pywrapcp.Solver,
        strategy: int,
        db_input: int,
        tile_h_out: Any,
        full_h_out: int,
    ) -> int:
        """
        Configure the inherited two-strategy search.

        Returns the output-buffering factor.
        """

        if strategy == 0:
            if db_input == 1:
                return 2

            solver.Add(tile_h_out == full_h_out)
            return 1

        # Strategy 1 only differs when the input is double-buffered.
        return 2 if db_input == 2 else 1

    def _symbolic_l3_memory(
        self,
        *,
        tile_h_in: Any,
        tile_h_out: Any,
        db_input: int,
        db_output: int,
    ) -> SymbolicPoolMemory:
        """Build symbolic L2 memory expressions for a height-only tile."""

        in_ch = self.HW_node.input_channels
        out_ch = self.HW_node.output_channels
        input_w = self.HW_node.input_dimensions[1]
        output_w = self.HW_node.output_dimensions[1]

        input_bits = (
            db_input
            * in_ch
            * tile_h_in
            * input_w
            * self.HW_node.input_activation_bits
        )
        output_bits = (
            db_output
            * out_ch
            * tile_h_out
            * output_w
            * self.HW_node.output_activation_bits
        )

        input_bytes = self._symbolic_bytes_for_bits(input_bits)
        output_bytes = self._symbolic_bytes_for_bits(output_bits)

        constants_bytes = self._constant_bytes_for_channels(out_ch)

        total = input_bytes + output_bytes + constants_bytes

        return SymbolicPoolMemory(
            input=input_bytes,
            output=output_bytes,
            constants=constants_bytes,
            fixed_overhead=0,
            total=total,
        )

    @staticmethod
    def _build_l3_objective(
        *,
        solver: pywrapcp.Solver,
        memory_total: Any,
        tile_h_in: Any,
        tile_h_out: Any,
    ) -> Any:
        """Preserve the inherited weighted L3 objective."""

        score = (
            memory_total
            + 200_000 * ((tile_h_out - 1) % 8)
            + 200_000 * ((tile_h_in - 1) % 4)
        )

        objective_expr = solver.IntVar(
            0,
            1_000_000_000_000,
            "objective",
        )
        solver.Add(objective_expr == score)
        return objective_expr

    @staticmethod
    def _solve_l3_problem(
        *,
        solver: pywrapcp.Solver,
        tile_h_in: Any,
        tile_h_out: Any,
        objective_expr: Any,
    ) -> tuple[int, int] | None:
        """Solve one L3 strategy and extract concrete heights."""

        ordered = [tile_h_in, tile_h_out]

        decision_builder = solver.Phase(
            ordered,
            solver.CHOOSE_FIRST_UNBOUND,
            solver.ASSIGN_MIN_VALUE,
        )
        objective = solver.Maximize(objective_expr, 1)

        collector = solver.LastSolutionCollector()
        for variable in ordered:
            collector.Add(variable)
        collector.AddObjective(objective_expr)

        solver.Solve(decision_builder, [objective, collector])

        if collector.SolutionCount() == 0:
            return None

        best = collector.SolutionCount() - 1
        return (
            collector.Value(best, tile_h_in),
            collector.Value(best, tile_h_out),
        )

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

    # ------------------------------------------------------------------
    # L2 -> L1
    # ------------------------------------------------------------------

    def get_tiling_pool2d_l2(
        self,
    ) -> tuple[list[int], list[int], list[int]]:
        """Find an L2->L1 pooling tile."""

        l1_memory = self._available_l1_memory()

        input_dimensions = self.HW_node.tiling_dimensions["L2"][
            "input_dimensions"
        ]
        output_dimensions = self.HW_node.tiling_dimensions["L2"][
            "output_dimensions"
        ]

        in_ch, input_h, input_w = input_dimensions
        out_ch, output_h, output_w = output_dimensions

        self._validate_channel_geometry(in_ch=in_ch, out_ch=out_ch)

        full_tile = PoolTileShape(
            channels=in_ch,
            h_in=input_h,
            w_in=input_w,
            h_out=output_h,
            w_out=output_w,
        )
        full_memory = self._concrete_l1_memory(
            tile=full_tile,
            double_buffering=1,
            fixed_overhead=0,
        )

        if full_memory.total <= l1_memory:
            return (
                [],
                list(input_dimensions),
                list(output_dimensions),
            )

        db = self.double_buffering
        ks = self.HW_node.kernel_shape
        strides = self.HW_node.strides
        pads = self.HW_node.pads

        solver = pywrapcp.Solver(
            "Pool2D_L2_L1",
            pywrapcp.Solver.DefaultSolverParameters(),
        )

        variables = self._create_l2_l1_variables(
            solver=solver,
            channels=in_ch,
            input_h=input_h,
            input_w=input_w,
            output_h=output_h,
            output_w=output_w,
            ks=ks,
        )

        self._add_l2_l1_constraints(
            solver=solver,
            variables=variables,
            ks=ks,
            strides=strides,
        )

        symbolic_memory = self._symbolic_l1_memory(
            tile_channels=variables["channels"],
            tile_h_in=variables["h_in"],
            tile_w_in=variables["w_in"],
            tile_h_out=variables["h_out"],
            tile_w_out=variables["w_out"],
            double_buffering=db,
            fixed_overhead=20,
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
                "Pool2D: no L2->L1 tile fits in {} bytes of L1.".format(
                    l1_memory
                )
            )

        tile = self._normalize_full_spatial_tile(
            tile=tile,
            input_h=input_h,
            input_w=input_w,
            ks=ks,
            strides=strides,
            pads=pads,
        )

        concrete_memory = self._concrete_l1_memory(
            tile=tile,
            double_buffering=db,
            fixed_overhead=20,
        )
        if concrete_memory.total > l1_memory:
            raise RuntimeError(
                "Pool2D solver returned an invalid tile: {} bytes required, "
                "{} bytes available. Breakdown: {}".format(
                    concrete_memory.total,
                    l1_memory,
                    concrete_memory,
                )
            )

        return (
            [],
            [tile.channels, tile.h_in, tile.w_in],
            [tile.channels, tile.h_out, tile.w_out],
        )

    @staticmethod
    def _validate_channel_geometry(*, in_ch: int, out_ch: int) -> None:
        if in_ch != out_ch:
            raise ValueError(
                "Pooling requires equal input/output channel counts, got "
                "{} and {}.".format(in_ch, out_ch)
            )

    @staticmethod
    def _create_l2_l1_variables(
        *,
        solver: pywrapcp.Solver,
        channels: int,
        input_h: int,
        input_w: int,
        output_h: int,
        output_w: int,
        ks: tuple[int, int],
    ) -> dict[str, Any]:
        return {
            "channels": solver.IntVar(1, channels, "tile_channels"),
            "h_in": solver.IntVar(ks[0], input_h, "tile_h_in"),
            "w_in": solver.IntVar(ks[1], input_w, "tile_w_in"),
            "h_out": solver.IntVar(1, output_h, "tile_h_out"),
            "w_out": solver.IntVar(1, output_w, "tile_w_out"),
        }

    def _add_l2_l1_constraints(
        self,
        *,
        solver: pywrapcp.Solver,
        variables: dict[str, Any],
        ks: tuple[int, int],
        strides: tuple[int, int],
    ) -> None:
        """Add pooling geometry and channel-alignment constraints."""

        solver.Add((variables["h_in"] - ks[0]) % strides[0] == 0)
        solver.Add((variables["w_in"] - ks[1]) % strides[1] == 0)

        alignment = int(
            8
            / min(
                self.HW_node.input_activation_bits,
                self.HW_node.output_activation_bits,
            )
        )
        solver.Add(variables["channels"] % alignment == 0)

        solver.Add(
            variables["h_out"] * strides[0]
            == variables["h_in"]
            - (ks[0] - 1)
            + (strides[0] - 1)
        )
        solver.Add(
            variables["w_out"] * strides[1]
            == variables["w_in"]
            - (ks[1] - 1)
            + (strides[1] - 1)
        )

    def _symbolic_l1_memory(
        self,
        *,
        tile_channels: Any,
        tile_h_in: Any,
        tile_w_in: Any,
        tile_h_out: Any,
        tile_w_out: Any,
        double_buffering: int,
        fixed_overhead: int,
    ) -> SymbolicPoolMemory:
        input_bits = (
            double_buffering
            * tile_channels
            * tile_h_in
            * tile_w_in
            * self.HW_node.input_activation_bits
        )
        output_bits = (
            double_buffering
            * tile_channels
            * tile_h_out
            * tile_w_out
            * self.HW_node.output_activation_bits
        )

        input_bytes = self._symbolic_bytes_for_bits(input_bits)
        output_bytes = self._symbolic_bytes_for_bits(output_bits)

        constants_bytes = (
            double_buffering
            * self._constant_bytes_for_channels(tile_channels)
        )

        total = (
            input_bytes
            + output_bytes
            + constants_bytes
            + fixed_overhead
        )

        return SymbolicPoolMemory(
            input=input_bytes,
            output=output_bytes,
            constants=constants_bytes,
            fixed_overhead=fixed_overhead,
            total=total,
        )

    def _concrete_l1_memory(
        self,
        *,
        tile: PoolTileShape,
        double_buffering: int,
        fixed_overhead: int,
    ) -> PoolMemoryBreakdown:
        input_bits = (
            double_buffering
            * tile.channels
            * tile.h_in
            * tile.w_in
            * self.HW_node.input_activation_bits
        )
        output_bits = (
            double_buffering
            * tile.channels
            * tile.h_out
            * tile.w_out
            * self.HW_node.output_activation_bits
        )

        return PoolMemoryBreakdown(
            input=self._bytes_for_bits(input_bits),
            output=self._bytes_for_bits(output_bits),
            constants=double_buffering
            * self._constant_bytes_for_channels(tile.channels),
            fixed_overhead=fixed_overhead,
        )

    def _constant_bytes_for_channels(self, channels: Any) -> Any:
        """Return quantization-constant storage for a channel tile."""

        constant_count = sum(
            name in {"l", "k"} for name in self.HW_node.constant_names
        )

        if constant_count == 0:
            return 0

        bits = (
            channels
            * constant_count
            * self.HW_node.constant_bits
        )
        return self._symbolic_bytes_for_bits(bits)

    @staticmethod
    def _build_l2_l1_objective(
        *,
        solver: pywrapcp.Solver,
        variables: dict[str, Any],
        memory_total: Any,
    ) -> Any:
        """Preserve the inherited weighted L2->L1 objective."""

        score = (
            10_000 * memory_total
            + 100 * variables["w_in"]
            + variables["h_in"]
            + 1_000_000 * variables["channels"]
        )

        objective_expr = solver.IntVar(
            0,
            1_000_000_000_000,
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
    ) -> PoolTileShape | None:
        ordered = [
            variables["channels"],
            variables["h_in"],
            variables["w_in"],
            variables["h_out"],
            variables["w_out"],
        ]

        decision_builder = solver.Phase(
            ordered,
            solver.CHOOSE_FIRST_UNBOUND,
            solver.ASSIGN_MIN_VALUE,
        )
        objective = solver.Maximize(objective_expr, 1)

        collector = solver.LastSolutionCollector()
        for variable in ordered:
            collector.Add(variable)
        collector.AddObjective(objective_expr)

        solver.Solve(decision_builder, [objective, collector])

        if collector.SolutionCount() == 0:
            return None

        best = collector.SolutionCount() - 1
        return PoolTileShape(
            channels=collector.Value(best, variables["channels"]),
            h_in=collector.Value(best, variables["h_in"]),
            w_in=collector.Value(best, variables["w_in"]),
            h_out=collector.Value(best, variables["h_out"]),
            w_out=collector.Value(best, variables["w_out"]),
        )

    def _normalize_full_spatial_tile(
        self,
        *,
        tile: PoolTileShape,
        input_h: int,
        input_w: int,
        ks: tuple[int, int],
        strides: tuple[int, int],
        pads: tuple[int, int, int, int],
    ) -> PoolTileShape:
        """
        Recompute output dimensions when a selected input tile spans a full axis.

        This includes padding only for a full spatial dimension.
        """

        h_in = min(tile.h_in, input_h)
        w_in = min(tile.w_in, input_w)
        h_out = tile.h_out
        w_out = tile.w_out

        if h_in == input_h:
            h_out = self._output_dimension(
                h_in,
                ks[0],
                strides[0],
                pads[0],
                pads[2],
            )

        if w_in == input_w:
            # Fixed defect from the original source: horizontal geometry must
            # use strides[1], not strides[0].
            w_out = self._output_dimension(
                w_in,
                ks[1],
                strides[1],
                pads[1],
                pads[3],
            )

        return PoolTileShape(
            channels=tile.channels,
            h_in=h_in,
            w_in=w_in,
            h_out=h_out,
            w_out=w_out,
        )
