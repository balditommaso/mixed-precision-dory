
from .tiler_conv2d import Tiler_Conv2D_PULP as Tiler_Conv2D
from .tiler_pool2d import Tiler_Pool2D_PULP as Tiler_Pool2D
from .tiler_add import Tiler_Add_PULP as Tiler_Add
from dory.Parsers.HW_node import HW_node


class Tiler_PULP:
    def __init__(
        self,
        HW_node: HW_node,
        previous_HW_node: HW_node,
        code_reserved_space: int,
        double_buffering: int = 2,
        num_cores: int = 8,
        input_HW_nodes=None,
    ) -> None:

        self.HW_node = HW_node

        # Keep this until all tilers have been migrated.
        self.previous_HW_node = previous_HW_node

        if input_HW_nodes is None:
            self.input_HW_nodes = (
                []
                if previous_HW_node is None
                else [previous_HW_node]
            )
        else:
            self.input_HW_nodes = list(input_HW_nodes)

        self.code_reserved_space = code_reserved_space
        self.double_buffering = double_buffering
        self.n_memory_levels = HW_node.HW_description["memory"]["levels"]
        self.num_cores = num_cores


    def get_tiling(self, level):
        # This function is used to create the tiling of either a convolutional layer or
        # a fully connected or a pooling layer. The relu is included automatically in conv/FC.
        if 'Conv' in self.HW_node.name or 'FullyConnected' in self.HW_node.name:
            return Tiler_Conv2D(self).get_tiling(level)
        elif 'Pool' in self.HW_node.name:
            return Tiler_Pool2D(self).get_tiling(level)
        elif 'Addition' in self.HW_node.name:
            return Tiler_Add(self).get_tiling(level)
        else:
            print("Not supported Layer.")
            return None
        
        