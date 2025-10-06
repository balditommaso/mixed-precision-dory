import numpy as np
from copy import deepcopy
from onnx import helper, TensorProto, numpy_helper
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from typing import *



class RecordOutScale(BaseTrasformation):
    """
    Record the out scale in the op. node
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        
    
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        iter_graph = deepcopy(graph)
        for node in iter_graph.node:
            # check operation which could have static parameters
            if node.op_type == "Quant" or len(node.input) < 2:
                continue
            
            if len(node.input) == 2:
                # out scale is given by the product of the scale 
                in_quant = model.find_producer(node.input[0])
                w_quant = model.find_producer(node.input[1])
                
                in_scale = model.get_initializer(in_quant.input[1])
                w_scale = model.get_initializer(w_quant.input[1])
                
                out_scale = np.prod(in_scale, w_scale)
            elif len(node.input) == 3:
                # extract the out scale from the bias
                b_quant = model.find_producer(node.input[2])
                
                out_scale = model.get_initializer(b_quant.input[1])
            else:
                self.warning_message(f"{node.name} has more than 3 inputs, not handled yet.")
                continue
            
            
            attr = helper.make_attribute(
                "out_scale",
                helper.make_tensor(
                    name=model.make_new_valueinfo_name(),
                    data_type=TensorProto.FLOAT,
                    dims=out_scale.shape,
                    vals=out_scale.flatten().astype(float)
                )
            )
            
            node.attribute.append(attr)
        
        return (model, False)
            

   