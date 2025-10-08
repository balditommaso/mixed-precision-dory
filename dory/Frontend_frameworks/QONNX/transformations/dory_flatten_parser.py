import numpy as np
from copy import deepcopy
from onnx import helper, numpy_helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from dory.Frontend_frameworks.QONNX.transformations.fold_static_quant import *
from qonnx.util.basic import get_by_name
from typing import *



class DoryFlattenParser(BaseTrasformation):
    """
    Adapter from QONNX flatten operation to DORY reshape.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)        
    
    
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        iter_graph = deepcopy(graph)
        for node in iter_graph.node:
            if node.op_type != "Flatten":
                continue
            
            in_shape = model.get_tensor_shape(node.input[0])
            
            if len(in_shape) < 2:
                self.error_message(f"{node.op_type} has unexpected out tensor shape.", ValueError)
                
            flatten_dim = np.prod(in_shape[1:])
            
            shape_arr = np.array([-1, flatten_dim], dtype=np.int64)
            shape_tensor = numpy_helper.from_array(shape_arr, model.make_new_valueinfo_name())
            graph.initializer.append(shape_tensor)

            reshape_node = helper.make_node(
                "Reshape",
                inputs=[node.input[0], shape_tensor.name],
                outputs=[node.output[0]]
            )
            
            graph.node.remove(node)
            graph.node.append(reshape_node)
            
        return (model, False)
            
            
            
            
            
            
            
            