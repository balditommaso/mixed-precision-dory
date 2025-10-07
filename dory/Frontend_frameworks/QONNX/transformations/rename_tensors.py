from copy import deepcopy
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from onnx import helper


class RenameTensorsSequentially(BaseTrasformation):
    
    def __init__(self, verbose=False):
        super().__init__(verbose)
    
    def apply(self, model: ModelWrapper):
        graph = model.graph

        rename_map = {}
        counter = 0

        # Go through nodes in order — this keeps graph execution order
        for node in graph.node:
            # Rename only input[0] if it exists
            if len(node.input) > 0:
                in_name = node.input[0]
                if in_name not in rename_map:
                    rename_map[in_name] = str(counter)
                    counter += 1
                    node.input[0] = rename_map[in_name]
                else:
                    node.input[0] = rename_map[in_name]

            # Rename only output[0]
            if len(node.output) > 0:
                out_name = node.output[0]
                if out_name not in rename_map:
                    rename_map[out_name] = str(counter)
                    counter += 1
                    node.output[0] = rename_map[out_name]
                else:
                    node.output[0] = rename_map[out_name]

        # Update value_info, inputs, outputs, and initializers
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            if vi.name in rename_map:
                vi.name = rename_map[vi.name]

        for init in graph.initializer:
            if init.name in rename_map:
                init.name = rename_map[init.name]
                
    

        return (model, False)
