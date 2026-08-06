from typing import Any



class PULPTilerCommon:
    """Small reusable helper mixin for PULP tilers."""

    def _available_l1_memory(self) -> int:
        hw = self.HW_node.HW_description
        num_cores = getattr(self, "num_cores", 8)
        return (
            hw["memory"]["L1"]["dimension"]
            - hw["HW specific parameters"]["accelerator core0 stack"]
            - (num_cores - 1)
            * hw["HW specific parameters"]["accelerator core1-7 stack"]
        )


    def _available_l2_memory(self) -> int:
        return (
            self.HW_node.HW_description["memory"]["L2"]["dimension"]
            - self.code_reserved_space
        )


    @staticmethod
    def _bytes_for_bits(bit_count: int) -> int:
        """Concrete ceil(bits / 8)."""
        return (bit_count + 7) // 8


    @staticmethod
    def _symbolic_bytes_for_bits(bit_count_expr: Any) -> Any:
        """Symbolic ceil(bits / 8) for OR-Tools expressions."""
        return (bit_count_expr + 7) // 8
    
    
    @staticmethod
    def _output_dimension(
        input_dimension: int,
        kernel_dimension: int,
        stride: int,
        pad_before: int,
        pad_after: int,
    ) -> int:
        """Compute the output size for a complete spatial axis."""

        return (
            input_dimension
            - (kernel_dimension - 1)
            + pad_before
            + pad_after
            + stride
            - 1
        ) // stride