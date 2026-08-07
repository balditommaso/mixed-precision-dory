from dataclasses import dataclass
from typing import Any
import numpy as np
from ortools.constraint_solver import pywrapcp
from dory.Hardware_targets.PULP.Common.Tiler.tiler_common import PULPTilerCommon



@dataclass(frozen=True)
class TileShape:
    """Concrete tile dimensions after the solver has produced a solution."""

    n_in: int
    n_out: int
    h_in: int
    w_in: int
    h_out: int
    w_out: int


@dataclass(frozen=True)
class MemoryBreakdown:
    """Concrete memory footprint of an L1 tile, in bytes."""

    bias: int = 0
    input: int = 0
    output: int = 0
    weights: int = 0
    constants: int = 0
    im2col: int = 0
    lut: int = 0
    unpacked_weights: int = 0
    fixed_overhead: int = 0

    @property
    def total(self) -> int:
        return sum(
            (
                self.bias,
                self.input,
                self.output,
                self.weights,
                self.constants,
                self.im2col,
                self.lut,
                self.unpacked_weights,
                self.fixed_overhead,
            )
        )


@dataclass(frozen=True)
class SymbolicMemory:
    """Symbolic OR-Tools expressions representing an L1 memory footprint."""

    bias: Any
    input: Any
    output: Any
    weights: Any
    constants: Any
    im2col: Any
    lut: Any
    unpacked_weights: Any
    fixed_overhead: Any
    total: Any
    
    

class Tiler_Conv2D_PULP(PULPTilerCommon):
    """
    Generate L3->L2 or L2->L1 tiling for a convolution-like layer.

    ``tiler`` is an existing DORY tiler object.  The original implementation
    copies all of its attributes into this object, and this refactor preserves
    that convention for compatibility.
    """
    def __init__(self, tiler):
        self.__dict__ = tiler.__dict__  # HACK avoid import loop
        
        
    def get_tiling(self, level: int) -> tuple[list[int]]:
        if level == 3:
            tiling = self.get_tiling_l3()

            out_ch_exceeds = self.HW_node.output_channels > tiling[0][0]
            in_dim_exceeds = self.HW_node.input_dimensions[0] > tiling[1][1]
            out_dim_exceeds = self.HW_node.output_dimensions[0] > tiling[2][1]
            if out_ch_exceeds and (in_dim_exceeds or out_dim_exceeds):
                raise ValueError(
                    "Convolution: simultaneous L3 tiling of weights and "
                    "input/output activations is not supported."
                )
            return tiling
        
        if level == 2:
            return self.get_tiling_l2()
        
        raise ValueError(
            "Invalid tiling level: expected 3 for L3->L2 or 2 for L2->L1."
        )


    def get_tiling_l3(self) -> tuple[list[int]]:
        """
        Find an L3->L2 tile.

        The inherited algorithm tries four buffering/tiling strategies in a
        fixed order.  The first strategy that has a feasible CP solution wins.
        This is therefore not one global optimization across all strategies.
        """
        l2_memory = self._available_l2_memory()
        input_l3 = self._input_from_l3()    
                
        if self._full_layer_l2_bytes() <= l2_memory and not input_l3:
            return self._full_layer_tile()
        
        ks = self.HW_node.kernel_shape
        inp_dim = self.HW_node.input_dimensions
        out_dim = self.HW_node.output_dimensions
        out_ch = self.HW_node.output_channels
        in_ch = self.HW_node.input_channels
        strides = self.HW_node.strides
        group = self.HW_node.group
        pads = self.HW_node.pads
        
        
        for strategy in range(4):
            solver = pywrapcp.Solver(
                "Conv2d_L3_L2",
                pywrapcp.Solver.DefaultSolverParameters(),
            )
            
            tile_n_out = solver.IntVar(1, out_ch, "tile_n_out")
            tile_h_out = solver.IntVar(1, out_dim[0], "tile_h_out")
            is_branch_output = bool(
                getattr(
                    self.HW_node,
                    "branch_out",
                    getattr(self.HW_node, "branch_output", 0),
                )
            )
            is_branch_change = bool(
                getattr(self.HW_node, "branch_change", 0)
            )

            if is_branch_output or is_branch_change:
                # The network template saves these tensors from L2 to external RAM.
                solver.Add(tile_h_out == out_dim[0])
            zero = solver.IntVar(0, 0, "zero")

            if not input_l3:
                tile_h_in = solver.IntVar(inp_dim[0], inp_dim[0], "tile_h_in")
                db_input = 1
            else:
                tile_h_in = solver.IntVar(ks[0], inp_dim[0], "tile_h_in")
                solver.Add((tile_h_in - ks[0]) % strides[0] == 0)
                db_input = 2
                
            db_weights, db_output = self._configure_l3_strategy(
                solver,
                strategy,
                db_input,
                tile_n_out,
                tile_h_out,
                out_ch,
                out_dim[0],
            )
            
            # L2 footprint.  The output buffer intentionally uses full
            # ``out_ch`` exactly as the original code did.  This is suspicious
            # when ``tile_n_out`` is smaller and should be verified against the
            # generated DMA/output implementation.
            input_bits = (
                in_ch
                * tile_h_in
                * inp_dim[1]
                * self.HW_node.input_activation_bits
            )
            output_bits = (
                out_ch
                * tile_h_out
                * out_dim[1]
                * self.HW_node.output_activation_bits
            )
            bits_per_output_channel_of_weights = (
                int(in_ch / group)
                * int(np.prod(ks))
                * self.HW_node.weight_bits
            )
            bias_bits_per_output_channel = self.HW_node.bias_bits * int(
                self.HW_node.bias_memory != 0
            )

            input_bytes = db_input * input_bits // 8
            output_bytes = db_output * output_bits // 8
            weight_bytes = (
                db_weights
                * tile_n_out
                * (
                    bits_per_output_channel_of_weights
                    + bias_bits_per_output_channel
                )
                // 8
            )
            
            constant_count, constant_bits = self._constant_bits_or_none()
            if constant_count == 0:
                constants_bytes = 0
            else:
                constants_bytes = (
                    int(db_weights)
                    * tile_n_out
                    * int(constant_count)
                    * int(constant_bits)
                ) // 8
                
            l2_footprint = sum(
                (input_bytes, output_bytes, weight_bytes, constants_bytes)
            )
            solver.Add(l2_footprint <= l2_memory)
            
            # Relate input and output tile heights when spatial tiling is used.
            if db_input == 2 and db_output == 2:
                solver.Add(
                    tile_h_out * strides[0]
                    == tile_h_in - (ks[0] - 1) + (strides[0] - 1)
                )
                
            if db_input == 2:
                produced_rows = (tile_h_in - ks[0] + strides[0]) // strides[0]
                solver.Add((out_dim[0] - zero) % produced_rows == 0)
                
            obj_expr = self._build_l3_objective(
                solver=solver,
                l2_footprint=l2_footprint,
                tile_n_out=tile_n_out,
                tile_h_in=tile_h_in,
                tile_h_out=tile_h_out,
                out_ch=out_ch,
                in_ch=in_ch,
                inp_dim=inp_dim,
                out_dim=out_dim,
                ks=ks,
                pads=pads,
                group=group,
            )
            
            solution = self._solve_l3_problem(
                solver,
                obj_expr,
                tile_n_out,
                tile_h_in,
                tile_h_out
            )
            
            if solution is not None:
                selected_n_out, selected_h_in, selected_h_out = solution
                return (
                    [selected_n_out, in_ch],
                    [in_ch, selected_h_in, inp_dim[1]],
                    [out_ch, selected_h_out, out_dim[1]],
                )
            
        raise ValueError("Conv2D: no feasible L3->L2 tiling was found.")                
                


    def get_tiling_l2(self) -> tuple[list[int]]:
        """Find an L2->L1 tile using the CP solver."""
        l1_memory = self._available_l1_memory()
        inp_dim, out_dim, in_mem, out_mem, h_in, h_out = (
            self._normalized_l2_dimensions()
        )
        
        out_ch = self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][0]
        in_ch = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][0]
        ks = self.HW_node.kernel_shape
        strides = self.HW_node.strides
        group = self.HW_node.group
        pads = self.HW_node.pads
        
        full_memory = self._full_l2_tile_memory(
            inp_dim, in_mem, out_mem, in_ch
        )
        
        if full_memory.total <= l1_memory:
            return (
                self.HW_node.tiling_dimensions["L2"]["weights_dimensions"],
                [
                    in_ch,
                    h_in,
                    self.HW_node.tiling_dimensions["L2"]["input_dimensions"][2],
                ],
                [
                    out_ch,
                    h_out,
                    self.HW_node.tiling_dimensions["L2"]["output_dimensions"][2],
                ],
            )
            
        db = self.double_buffering
        
        solver = pywrapcp.Solver(
            "Conv2d_L2_L1",
            pywrapcp.Solver.DefaultSolverParameters()
        )
        variables = self._create_l2_l1_variables(
            solver, in_ch, out_ch, inp_dim, out_dim, ks
        )
        self._add_l2_l1_geometry_constraints(
            solver=solver,
            v=variables,
            inp_dim=inp_dim,
            ks=ks,
            strides=strides,
            pads=pads,
            group=group,
        )
        self._add_l2_l1_backend_constraints(
            solver, variables, inp_dim, out_dim, in_ch, ks, strides, pads, group
        )
        
        symbolic_memory = self._symbolic_l1_memory(
            tile_n_in=variables["n_in"],
            tile_n_out=variables["n_out"],
            tile_h_in=variables["h_in"],
            tile_w_in=variables["w_in"],
            tile_h_out=variables["h_out"],
            tile_w_out=variables["w_out"],
            double_buffering=db,
            bias_bytes=self.HW_node.tiling_dimensions["L2"]["bias_memory"],
            fixed_overhead=40,
        )
        solver.Add(symbolic_memory.total <= l1_memory)
        
        obj_expr = self._build_l2_l1_objective(
            solver,
            variables,
            symbolic_memory.total,
            out_ch,
            out_dim,
            group,
        )
        solution = self._solve_l2_l1_problem(solver, variables, obj_expr)
        if solution is None:
            raise ValueError(
            "Conv2D: no L2->L1 tile found.\n"
            + self._memory_failure_report(
                full_memory,
                l1_memory,
            )
        )
            
        tile = self._normalized_full_spatial_tile(
            solution, inp_dim, ks, strides, pads
        )
        
        concrete_memory = self._concrete_l1_memory(
            tile=tile,
            double_buffering=db,
            bias_bytes=self.HW_node.tiling_dimensions["L2"]["bias_memory"],
            fixed_overhead=40,
        )
        if concrete_memory.total > l1_memory:
            raise RuntimeError(
                "{}: Solver returned a tile that exceeds L1 according to the "
                "concrete model: {} > {} bytes. Breakdown: {}".format(
                    self.HW_node.name, concrete_memory.total, l1_memory, concrete_memory
                )
            )

        return (
            [tile.n_out, tile.n_in],
            [tile.n_in, tile.h_in, tile.w_in],
            [tile.n_out, tile.h_out, tile.w_out],
        )
        
        
    def _symbolic_l1_memory(
        self,
        tile_n_in: pywrapcp.Solver.IntVar,
        tile_n_out: pywrapcp.Solver.IntVar,
        tile_h_in: pywrapcp.Solver.IntVar,
        tile_w_in: pywrapcp.Solver.IntVar,
        tile_h_out: pywrapcp.Solver.IntVar,
        tile_w_out: pywrapcp.Solver.IntVar,
        double_buffering: int,
        bias_bytes: int,
        fixed_overhead: int = 40,
    ) -> SymbolicMemory:
        """
        Build OR-Tools expressions for all L1 memory components.

        Parameters such as ``tile_n_in`` are symbolic ``IntVar`` objects.
        No Python branching may depend on their values.  Branching here is
        allowed only on ordinary, already-known layer properties such as
        ``group`` or ``weight_bits``.
        """
        ks = self.HW_node.kernel_shape
        pads = self.HW_node.pads
        group = self.HW_node.group
        cores = self.num_cores
        precision_parallelism = self._precision_parallelism()
        
        input_bytes = double_buffering * (
            tile_n_in
            * tile_h_in
            * tile_w_in
            * self.HW_node.input_activation_bits
        ) // 8
        output_bytes = double_buffering * (
            tile_n_out
            * tile_h_out
            * tile_w_out
            * self.HW_node.output_activation_bits
        ) // 8
        
        if group == 1:
            weight_bytes = double_buffering * (
                tile_n_in
                * tile_n_out
                * int(np.prod(ks))
                * self.HW_node.weight_bits
            ) // 8
            im2col_bytes = 2 * cores * int(np.prod(ks)) * tile_n_in
            unpacked_weight_bytes = 0
        else:
            weight_bytes = double_buffering * (
                tile_n_in * int(np.prod(ks)) * self.HW_node.weight_bits
            ) // 8
            im2col_bytes = (
                cores
                * (ks[0] * (tile_n_in + pads[0] + pads[2]) + ks[0])
                * precision_parallelism
            )
            unpacked_weight_bytes = 0
            if self.HW_node.weight_bits != 8:
                unpacked_weight_bytes = (
                    double_buffering
                    * 8
                    * 8
                    * int(np.prod(ks))
                    * precision_parallelism
                )
        
        if "FullyConnected" in self.HW_node.name:
            im2col_bytes = 0
            
        constant_count, constant_bits = self._constant_bits_or_none()

        if constant_count == 0:
            constants_bytes = 0
        else:
            constants_bytes = (
                double_buffering
                * tile_n_out
                * constant_count
                * constant_bits
            ) // 8

        lut_bytes = self._lut_bytes()
        total = sum(
            (
                bias_bytes,
                input_bytes,
                output_bytes,
                weight_bytes,
                constants_bytes,
                im2col_bytes,
                lut_bytes,
                unpacked_weight_bytes,
                fixed_overhead,
            )
        )

        return SymbolicMemory(
            bias=bias_bytes,
            input=input_bytes,
            output=output_bytes,
            weights=weight_bytes,
            constants=constants_bytes,
            im2col=im2col_bytes,
            lut=lut_bytes,
            unpacked_weights=unpacked_weight_bytes,
            fixed_overhead=fixed_overhead,
            total=total,
        )
        

    @staticmethod
    def _configure_l3_strategy(
        solver: pywrapcp.Solver,
        strategy: int,
        db_input: int,
        tile_n_out: pywrapcp.Solver.IntVar,
        tile_h_out: pywrapcp.Solver.IntVar,
        out_ch: int,
        full_h_out: int,
    ) -> tuple[int, int]:
        """Configure one inherited L3 buffering strategy.

        Returns ``(db_weights, db_output)``.
        """

        if strategy == 0:
            if db_input == 1:
                solver.Add(tile_h_out == full_h_out)
                return 2, 1
            solver.Add(tile_h_out == full_h_out)
            solver.Add(tile_n_out == out_ch)
            return 1, 1

        if strategy == 1:
            solver.Add(tile_n_out == out_ch)
            return 1, 2

        if strategy == 2:
            if db_input == 1:
                return 2, 2
            solver.Add(tile_h_out == full_h_out)
            return 2, 1

        return 2, 2


    @staticmethod
    def _create_l2_l1_variables(
        solver: pywrapcp.Solver,
        in_ch: int,
        out_ch: int,
        inp_dim: tuple[int],
        out_dim: tuple[int],
        ks: tuple[int]
    ) -> dict[str, pywrapcp.Solver.IntVar]:
        return {
            "n_in": solver.IntVar(1, in_ch, "tile_n_in"),
            "n_out": solver.IntVar(1, out_ch, "tile_n_out"),
            "h_in": solver.IntVar(ks[0], inp_dim[0], "tile_h_in"),
            "w_in": solver.IntVar(ks[1], inp_dim[1], "tile_w_in"),
            "h_out": solver.IntVar(1, out_dim[0], "tile_h_out"),
            "w_out": solver.IntVar(1, out_dim[1], "tile_w_out"),
            "zero": solver.IntVar(0, 0, "zero"),
        }
        
        
    def _build_l3_objective(
        self,
        solver: pywrapcp.Solver,
        l2_footprint: int,
        tile_n_out: pywrapcp.Solver.IntVar,
        tile_h_in: pywrapcp.Solver.IntVar,
        tile_h_out: pywrapcp.Solver.IntVar,
        out_ch: int,
        in_ch: int,
        inp_dim: list[int],
        out_dim: list[int],
        ks: list[int],
        pads: list[int],
        group: int,
    ) -> pywrapcp.Solver.IntVar:
        """
        Build the inherited L3 heuristic objective.

        The objective rewards L2 utilization, a tile that also fits in L1,
        backend-friendly dimensions, and exact divisibility.  It is a weighted
        sum rather than a true lexicographic optimization.
        """

        l1_memory = self._available_l1_memory()
        cores = self.num_cores
        precision_parallelism = self._precision_parallelism()

        input_l1 = (
            in_ch
            * tile_h_in
            * inp_dim[1]
            * self.HW_node.input_activation_bits
            // 8
        )
        output_l1 = (
            tile_n_out
            * tile_h_out
            * out_dim[1]
            * self.HW_node.output_activation_bits
            // 8
        )

        if group == 1:
            weight_l1 = (
                in_ch
                * tile_n_out
                * int(np.prod(ks))
                * self.HW_node.weight_bits
                // 8
            )
            im2col_l1 = 2 * cores * int(np.prod(ks)) * in_ch
            unpacked_weights_l1 = 0
        else:
            weight_l1 = (
                in_ch
                * int(np.prod(ks))
                * self.HW_node.weight_bits
                // 8
            )
            im2col_l1 = (
                cores
                * (ks[0] * (in_ch + pads[0] + pads[2]) + ks[0])
                * precision_parallelism
            )
            unpacked_weights_l1 = 0
            if self.HW_node.weight_bits != 8:
                # Preserve the original L3 objective's formula exactly.
                unpacked_weights_l1 = (
                    32
                    * 8
                    * 8
                    * int(np.prod(ks))
                    * precision_parallelism
                )

        if "FullyConnected" in self.HW_node.name:
            im2col_l1 = 0

        constant_count, constant_bits = self._constant_bits_or_none()

        if constant_count == 0:
            constants_l1 = 0
        else:
            constants_l1 = (
                tile_n_out
                * constant_count
                * constant_bits
            ) // 8
            
        l1_footprint = sum(
            (
                input_l1,
                output_l1,
                weight_l1,
                constants_l1,
                im2col_l1,
                self._lut_bytes(),
                unpacked_weights_l1,
                20,
            )
        )

        objective_expr = solver.IntVar(0, 100_000_000_000_000, "objective")
        inherited_score = (
            l2_footprint
            + 100_000_000 * (l1_footprint < l1_memory)
            + 100_000 * ((tile_h_out - 1) % 8)
            + 100_000 * (tile_n_out % 4 == 0)
            + 1_000_000 * ((out_ch - tile_n_out) % tile_n_out == 0)
            + 1_000_000 * ((out_dim[0] - tile_h_out) % tile_h_out == 0)
            + 100_000 * (tile_h_out == out_dim[0])
            + 10_000 * (tile_n_out == out_ch)
        )
        solver.Add(objective_expr == inherited_score)
        return objective_expr


    @staticmethod
    def _solve_l3_problem(
        solver: pywrapcp.Solver,
        objective_expr: pywrapcp.Solver.IntVar,
        tile_n_out: pywrapcp.Solver.IntVar,
        tile_h_in: pywrapcp.Solver.IntVar,
        tile_h_out: pywrapcp.Solver.IntVar,
    ) -> tuple[int]:
        """Solve one L3 strategy and return concrete integer values."""

        objective = solver.Maximize(objective_expr, 1)
        decision_builder = solver.Phase(
            [tile_n_out, tile_h_in, tile_h_out],
            solver.CHOOSE_FIRST_UNBOUND,
            solver.ASSIGN_MIN_VALUE,
        )
        collector = solver.LastSolutionCollector()
        for variable in (tile_n_out, tile_h_in, tile_h_out):
            collector.Add(variable)
        collector.AddObjective(objective_expr)
        solver.Solve(decision_builder, [objective, collector])

        if collector.SolutionCount() == 0:
            return None

        best = collector.SolutionCount() - 1
        return (
            collector.Value(best, tile_n_out),
            collector.Value(best, tile_h_in),
            collector.Value(best, tile_h_out),
        )
        
        
    @staticmethod
    def _build_l2_l1_objective(
        solver: pywrapcp.Solver,
        v: dict[str, pywrapcp.Solver.IntVar],
        memory_total: int,
        out_ch: int,
        out_dim: tuple[int],
        group: int,
    ) -> pywrapcp.Solver.IntVar:
        """
        Build the inherited weighted heuristic objective.

        This is intentionally unchanged in meaning.  It should be replaced by
        named, bounded metrics or sequential optimization only after regression
        tests capture current tile choices.
        """
        zero = v["zero"]
        if group == 1:
            score = (
                2_000_000 * ((v["h_out"] - 1) % 8)
                + 3_000_000 * ((v["w_out"] - 1) % 2)
                + 1_000_000 * ((v["n_out"] - 1) % 4)
                + 1_000_000 * (v["w_out"] * v["h_out"] >= 16)
                + memory_total
                + 100_000 * v["n_out"]
                + 10_000 * ((out_ch - zero - 1) % v["n_out"])
                + 10_000 * (((out_ch - zero - 1) % v["n_out"]) % 4)
                + 20_000 * (((out_dim[0] - zero - 1) % v["h_out"]) % 8)
                + 30_000 * (((out_dim[1] - zero - 1) % v["w_out"]) % 2)
            )
        else:
            score = (
                10_000 * (v["n_out"] > 7)
                + 20_000 * ((v["n_out"] - 1) % 16)
                + 10_000 * (v["h_out"] % 4 == 0)
                + memory_total
                + 1_000 * v["w_out"]
                + 1_000 * v["h_out"]
                + 100 * ((out_dim[0] - zero - 1) % v["h_out"])
                + 100 * ((out_dim[1] - zero - 1) % v["w_out"])
                + 100 * (((out_ch - zero - 1) % v["n_out"]) > 7)
                + 100 * (((out_dim[0] - zero - 1) % v["h_out"]) % 4)
            )
        
        obj_expr = solver.IntVar(0, 1_000_000_000_000, "objective")
        solver.Add(obj_expr == score)
        return obj_expr
        
    
    @staticmethod
    def _solve_l2_l1_problem(
        solver: pywrapcp.Solver, 
        variables: dict[str, Any], 
        objective_expr: pywrapcp.Solver.IntVar
    ) -> TileShape:
        """Run the CP search and return a concrete ``TileShape``."""
        ordered = [
            variables["n_in"],
            variables["n_out"],
            variables["h_in"],
            variables["h_out"],
            variables["w_in"],
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
        return TileShape(
            n_in=collector.Value(best, variables["n_in"]),
            n_out=collector.Value(best, variables["n_out"]),
            h_in=collector.Value(best, variables["h_in"]),
            w_in=collector.Value(best, variables["w_in"]),
            h_out=collector.Value(best, variables["h_out"]),
            w_out=collector.Value(best, variables["w_out"]),
        )
    
        
    @staticmethod
    def _add_l2_l1_geometry_constraints(
        solver: pywrapcp.Solver,
        v: dict[str, pywrapcp.Solver.IntVar],
        inp_dim: tuple[int],
        ks: tuple[int],
        strides: tuple[int],
        pads: tuple[int],
        group: int,
    ) -> None:
        """Relate input and output tile geometry."""
        if group == 1 or (inp_dim[0] > 32 and inp_dim[1] > 32):
            solver.Add((v["h_in"] - ks[0]) % strides[0] == 0)
            
        if group > 1:
            solver.Add(v["n_in"] == v["n_out"])
            
        if group == 1:
            # Padding is applied only when a tile spans the full dimension.
            solver.Add(
                v["h_out"] * strides[0]
                == v["h_in"]
                - (ks[0] - 1)
                + (strides[0] - 1)
                + (pads[0] + pads[2]) * (v["h_in"] == inp_dim[0])
            )
            solver.Add(
                v["w_out"] * strides[1]
                == v["w_in"]
                - (ks[1] - 1)
                + (strides[1] - 1)
                + (pads[1] + pads[3]) * (v["w_in"] == inp_dim[1])
            )
            
            
    def _add_l2_l1_backend_constraints(
        self,
        solver: pywrapcp.Solver,
        v: dict[str, pywrapcp.Solver.IntVar],
        inp_dim: tuple[int],
        out_dim: tuple[int],
        in_ch: int,
        ks: tuple[int],
        strides: tuple[int],
        pads: tuple[int],
        group: int,
    ) -> None:
        """Add PULP-NN implementation restrictions and channel alignment."""

        if group > 1:
            if inp_dim[0] <= 32 and inp_dim[1] <= 32:
                # Small depthwise layers are not spatially tiled.
                solver.Add(v["h_in"] == inp_dim[0])
                solver.Add(v["w_in"] == inp_dim[1])
                solver.Add(v["h_out"] == out_dim[0])
                solver.Add(v["w_out"] == out_dim[1])
            else:
                solver.Add(
                    v["h_out"] * strides[0]
                    == v["h_in"]
                    - (ks[0] - 1)
                    + (v["h_in"] % inp_dim[0] == 0) * (pads[0] + pads[2])
                    + (strides[0] - 1)
                )
                # INHERITED BACKEND LIMIT: depthwise width is kept full.
                solver.Add(v["w_in"] == inp_dim[1])
                solver.Add(v["w_out"] == out_dim[1])

            solver.Add(v["n_in"] % self._precision_parallelism() == 0)

        if group == 1:
            # Standard convolution does not tile input channels in this backend.
            solver.Add(v["n_in"] == int(in_ch))

        solver.Add(v["n_out"] % self._precision_parallelism() == 0)


    def _full_l2_tile_memory(
        self,
        inp_dim: tuple[int],
        in_mem: int,
        out_mem: int,
        in_ch: int,
    ) -> MemoryBreakdown:
        """Build the inherited concrete early-fit footprint for the full L2 tile."""
        ks = self.HW_node.kernel_shape
        pads = self.HW_node.pads
        group = self.HW_node.group
        cores = self.num_cores
        precision_parallelism = self._precision_parallelism()
        
        if group == 1:
            # TODO: check implementation
            im2col = int(
                2
                * cores
                * int(np.prod(ks))
                * in_ch
                * self.HW_node.input_activation_bits
                / 8
            )
            unpacked_weights = 0
        else:
            im2col = (
                cores
                * (ks[0] * (inp_dim[0] + pads[0] + pads[2]) + ks[0])
                * precision_parallelism
            )
            unpacked_weights = 8 * int(np.prod(ks)) * precision_parallelism
            if self.HW_node.weight_bits == 8:
                unpacked_weights = 0
        
        if "FullyConnected" in self.HW_node.name:
            im2col = 0

        return MemoryBreakdown(
            bias=self.HW_node.tiling_dimensions["L2"]["bias_memory"],
            input=int(in_mem),
            output=int(out_mem),
            weights=self.HW_node.tiling_dimensions["L2"]["weight_memory"],
            constants=self.HW_node.tiling_dimensions["L2"]["constants_memory"],
            im2col=int(im2col),
            lut=self._lut_bytes(),
            unpacked_weights=int(unpacked_weights),
            fixed_overhead=0,
        )


    def _available_l2_memory(self) -> int:
        """Return L2 bytes available after reserving generated code space."""
        return (
            self.HW_node.HW_description["memory"]["L2"]["dimension"]
            - self.code_reserved_space
        )
        
        
    def _full_layer_l2_bytes(self) -> int:
        """Return the already-computed full-layer L2 footprint."""

        return sum(
            (
                self.HW_node.input_activation_memory,
                self.HW_node.output_activation_memory,
                self.HW_node.weight_memory,
                self.HW_node.bias_memory,
                self.HW_node.constants_memory,
            )
        )

    def _full_layer_tile(self):
        """Return the original full-layer tuple format."""

        return (
            [self.HW_node.output_channels, self.HW_node.input_channels],
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


    def _input_from_l3(self) -> bool:
        producer = (
            self.input_HW_nodes[0]
            if self.input_HW_nodes
            else None
        )

        if producer is None:
            self.HW_node.L3_input = 0
            return False

        l3_out = producer.tiling_dimensions["L3"][
            "output_dimensions"
        ]

        l2_out = producer.tiling_dimensions["L2"][
            "output_dimensions"
        ]

        comes_from_l3 = (
            l2_out is not None
            and l3_out != l2_out
        )

        self.HW_node.L3_input = int(comes_from_l3)
        return comes_from_l3
                

    def _available_l1_memory(self) -> int:
        hw = self.HW_node.HW_description
        return (
            hw["memory"]["L1"]["dimension"]
            - hw["HW specific parameters"]["accelerator core0 stack"]
            - (self.num_cores - 1)
            * hw["HW specific parameters"]["accelerator core1-7 stack"]
        )
    
    
    def _concrete_l1_memory(
        self,
        tile: TileShape,
        double_buffering: int,
        bias_bytes: int,
        fixed_overhead: int = 40
    ) -> MemoryBreakdown:
        """
        Evaluate L1 memory for one *known* tile using Python integers.

        This method must never receive an OR-Tools ``IntVar``.  It is useful
        for early-fit checks, diagnostics, and validating a solver result.
        """
        ks = self.HW_node.kernel_shape
        pads = self.HW_node.pads
        group = self.HW_node.group
        cores = self.num_cores
        precision_parallelism = self._precision_parallelism()

        input_bits = (
            tile.n_in
            * tile.h_in
            * tile.w_in
            * self.HW_node.input_activation_bits
        )
        output_bits = (
            tile.n_out
            * tile.h_out
            * tile.w_out
            * self.HW_node.output_activation_bits
        )
        
        input_bytes = double_buffering * self._bits_to_bytes(input_bits)
        output_bytes = double_buffering * self._bits_to_bytes(output_bits)
        
        if group == 1:
            weight_bits = (
                tile.n_in
                * tile.n_out
                * int(np.prod(ks))
                * self.HW_node.weight_bits
            )
            weight_bytes = double_buffering * self._bits_to_bytes(weight_bits)

            # INHERITED ASSUMPTION: im2col is byte-addressed independently of
            # activation precision in the symbolic solver path.
            im2col_bytes = 2 * cores * int(np.prod(ks)) * tile.n_in
            unpacked_weight_bytes = 0
        else:
            weight_bits = (
                tile.n_in * int(np.prod(ks)) * self.HW_node.weight_bits
            )
            weight_bytes = double_buffering * self._bits_to_bytes(weight_bits)
            im2col_bytes = (
                cores
                * (ks[0] * (tile.n_in + pads[0] + pads[2]) + ks[0])
                * precision_parallelism
            )
            unpacked_weight_bytes = 0
            if self.HW_node.weight_bits != 8:
                unpacked_weight_bytes = (
                    double_buffering
                    * 8
                    * 8
                    * int(np.prod(ks))
                    * precision_parallelism
                )
                
        if "FullyConnected" in self.HW_node.name:
            im2col_bytes = 0

        constant_count, constant_bits = self._constant_bits_or_none()

        if constant_count == 0:
            constants_bytes = 0
        else:
            constants_bits = (
                int(tile.n_out)
                * constant_count
                * constant_bits
            )
            constants_bytes = (
                int(double_buffering)
                * self.__to_bytes(constants_bits)
            )

        return MemoryBreakdown(
            bias=bias_bytes,
            input=input_bytes,
            output=output_bytes,
            weights=weight_bytes,
            constants=constants_bytes,
            im2col=im2col_bytes,
            lut=self._lut_bytes(),
            unpacked_weights=unpacked_weight_bytes,
            fixed_overhead=fixed_overhead,
        )
        
        
    def _normalized_l2_dimensions(self) -> tuple[int]:
        """
        Return mutable local L2 dimensions and memories.

        This isolates the inherited rescaling logic used when an L3 tile are
        bigger than the corresponding L2 tile.
        """
        inp_dim = list(self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1:])
        out_dim = list(self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1:])
        in_mem = self.HW_node.tiling_dimensions["L2"]["input_activation_memory"]
        out_mem = self.HW_node.tiling_dimensions["L2"]["output_activation_memory"]
        h_in = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1]
        h_out = self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1]
        
        ks = self.HW_node.kernel_shape
        strides = self.HW_node.strides
        
        if self.n_memory_levels > 2:
            l3_out_h = self.HW_node.tiling_dimensions["L3"]["output_dimensions"][1]
            l2_out_h = self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1]
            if l3_out_h > l2_out_h:
                scaled_h_in = h_out * strides[0] + (ks[0] - 1) - (strides[0] - 1)
                inp_dim[0] = scaled_h_in
                in_mem = int(in_mem / h_in * scaled_h_in)
                h_in = scaled_h_in
            
            l3_in_h = self.HW_node.tiling_dimensions["L3"]["input_dimensions"][1]
            if l3_in_h > h_in:
                scaled_h_out = int(
                    np.floor(
                        (h_in - (ks[0] - 1) + (strides[0] - 1))
                        / strides[0]
                    )
                )
                out_dim[0] = scaled_h_out
                out_mem = int(out_mem / h_out * scaled_h_out)
                h_out = scaled_h_out
                
        if "Addition" not in self.HW_node.name and "Pool" not in self.HW_node.name:
            output_channels = self.HW_node.tiling_dimensions["L2"][
                "output_dimensions"
            ][0]
            weight_output_channels = self.HW_node.tiling_dimensions["L2"][
                "weights_dimensions"
            ][0]
            out_mem = int(out_mem / output_channels * weight_output_channels)

        return inp_dim, out_dim, in_mem, out_mem, h_in, h_out
    
    
    def _normalized_full_spatial_tile(
        self,
        tile: TileShape,
        inp_dim: tuple[int],
        ks: tuple[int],
        strides: tuple[int],
        pads: tuple[int],
    ) -> TileShape:
        """
        Recompute outputs when the selected input tile spans a full axis.
        """
        h_in, h_out = tile.h_in, tile.h_out
        w_in, w_out = tile.w_in, tile.w_out

        if h_in >= inp_dim[0]:
            h_in = inp_dim[0]
            h_out = self._output_dimension(
                h_in, ks[0], strides[0], pads[0], pads[2]
            )

        if w_in >= inp_dim[1]:
            w_in = inp_dim[1]
            w_out = self._output_dimension(
                w_in, ks[1], strides[1], pads[1], pads[3]
            )

        return TileShape(
            n_in=tile.n_in,
            n_out=tile.n_out,
            h_in=h_in,
            w_in=w_in,
            h_out=h_out,
            w_out=w_out,
        )

    
    def _bn_constant_count(self) -> int:
        """
        Count per-output-channel batch-normalization constants.
        """
        return sum(
            name in ("l", "k") for name in self.HW_node.constant_names
        )
        
        
    def _constant_bits_or_none(self) -> tuple[int, int | None]:
        count = self._bn_constant_count()

        if count == 0:
            return 0, None

        if self.HW_node.constant_bits is None:
            raise ValueError(
                f"Layer {self.HW_node.name!r} contains BN constants "
                "but constant_bits is None."
            )

        return count, int(self.HW_node.constant_bits)
        
         
    def _lut_bytes(self) -> int:
        """Return storage in bytes for the magnitude-only LUT."""

        if self.HW_node.implementation != "lut":
            return 0

        input_bits = self.HW_node.input_activation_bits
        weight_bits = self.HW_node.weight_bits

        # Replace this with the actual signedness attribute in your model.
        input_is_signed = self.HW_node.input_activation_type == "int"

        if input_is_signed:
            input_entries = (1 << (input_bits - 1)) + 1
        else:
            input_entries = 1 << input_bits

        # Weights are signed: magnitudes range from 0 to 2^(bits-1).
        weight_entries = (1 << (weight_bits - 1)) + 1
        entries = input_entries * weight_entries

        return entries * 2
    
    
    def _precision_parallelism(self) -> int:
        """Return the inherited channel-alignment factor for sub-byte kernels."""

        minimum_bits = min(
            self.HW_node.input_activation_bits,
            self.HW_node.output_activation_bits,
            self.HW_node.weight_bits,
        )
        return int(8 / minimum_bits)
    
    
    @staticmethod
    def _output_dimension(
        input_dimension: int,
        kernel_dimension: int,
        stride: int,
        pad_before: int,
        pad_after: int,
    ) -> int:
        """Compute a full-tensor output dimension with integer convolution geometry."""

        return (
            input_dimension
            - (kernel_dimension - 1)
            + pad_before
            + pad_after
            + stride
            - 1
        ) // stride

    
    
    @staticmethod
    def _bits_to_bytes(bit_count: int):
        """Return the number of bytes required to store bit_count bits."""
        return (bit_count + 7) // 8    
    
    
    def _memory_failure_report(
        self,
        memory: MemoryBreakdown,
        available: int,
    ) -> str:
        components = {
            "input": memory.input,
            "output": memory.output,
            "weights": memory.weights,
            "bias": memory.bias,
            "constants": memory.constants,
            "im2col": memory.im2col,
            "lut": memory.lut,
            "unpacked_weights": memory.unpacked_weights,
            "fixed_overhead": memory.fixed_overhead,
        }

        lines = [
            f"Layer: {self.HW_node.name}",
            f"Available L1: {available} bytes",
            f"Required by full tile: {memory.total} bytes",
            f"Excess: {max(0, memory.total - available)} bytes",
            "Memory breakdown:",
        ]

        for name, size in sorted(
            components.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            percentage = 100.0 * size / memory.total if memory.total else 0.0
            lines.append(
                f"  {name:<20}: {size:>8} bytes ({percentage:5.1f}%)"
            )

        return "\n".join(lines)
            
            