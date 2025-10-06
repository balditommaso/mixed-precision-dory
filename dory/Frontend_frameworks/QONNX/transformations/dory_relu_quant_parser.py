import numpy as np
from copy import deepcopy
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from qonnx.util.basic import get_by_name
from typing import *



class DoryActQuantParser(BaseTrasformation):
    """
    Look for ReLU node followed by Quant nodes and convert it to
    Dory expected format.
    """
    
    def __init__(self, delta: int, verbose: bool = False):
        super().__init__(verbose)
        self.delta = delta
        
    
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        iter_graph = deepcopy(graph)
        for node in iter_graph.node:
            if node.op_type != "Relu":
                continue
            
            quant_node = model.find_consumer(node.output[0])
            if quant_node is None:
                self.warning_message(f"{node.name} is not followed by a Quant node.")
                continue
            
            prev_node_out = node.input[0]
            
            # Mul node with the scale function multiplied by the scale
            
            
            
        return (model, False)
            
            
            
            
            
            
            
            