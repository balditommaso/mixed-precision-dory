import numpy as np
from copy import deepcopy
from onnx import helper, numpy_helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper, NodeProto
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from dory.Frontend_frameworks.QONNX.transformations.fold_static_quant import *
from dory.Frontend_frameworks.QONNX.transformations.dory_quant_parser import get_prev_out_scale
from qonnx.util.basic import get_by_name
from typing import *

MAX_DEPTH = 5



def merge_add_quant(mw: ModelWrapper, node: NodeProto, delta: int = 16) -> None:
    # check if the expected pattern is correct
    inp1, inp2 = mw.find_producer(node.input[0]), mw.find_producer(node.input[1])
    assert inp1 is not None and inp1.op_type == "Quant", f"{inp1.name} should be a `Quant` node"
    assert inp2 is not None and inp2.op_type == "Quant", f"{inp2.name} should be a `Quant` node"
    
    quant_node = None
    
    while quant_node is None:
        successor = mw.find_direct_successors(node)
        assert successor and len(successor) == 1 
        
        if successor[0].op_type == "Quant":
            quant_node = successor[0]
    
        elif successor[0].op_type == "Relu":  
            relu_node = successor[0]
            relu_input = relu_node.input[0]
            next_node = mw.find_consumer(relu_node.output[0])
            next_node.input[0] = relu_input
            mw.graph.node.remove(relu_node)
        else:
            raise NotImplementedError("Unexpected op. after Add.")
    
    # merge quant info into the add node  
    out_scale = get_prev_out_scale(mw, quant_node)
    quant_scale = mw.get_initializer(quant_node.input[1])
    rounding_mode = get_by_name(quant_node.attribute, "rounding_mode").s.decode("utf-8")
    round_fx = resolve_rounding_mode(rounding_mode)
    
    out_shape = mw.get_tensor_shape(quant_node.input[0])
    C_out = out_shape[1] if len(out_shape) > 1 else out_shape[0]
    
    M = round_fx(out_scale / quant_scale * 2 ** delta).astype(np.float32)
    
    zeropt = mw.get_initializer(quant_node.input[2])
    if zeropt is not None and not np.all(zeropt == 0.):
        Z = round_fx(zeropt * out_scale * 2 ** delta)
        if np.isscalar(Z) or np.size(Z) == 1:
            Z = np.full((C_out, 1, 1), float(Z), dtype=np.float32)
        else:
            Z = np.reshape(Z, (C_out, 1, 1)).astype(np.float32)
    else:
        Z = 0
                
    out_bit_width = int(mw.get_initializer(quant_node.input[3]))
    
    for prefix in ["in1", "in2", "out"]:
        rq_attr = helper.make_attribute(
            f"{prefix}_rq", 
            1
        )
        add_attr = helper.make_attribute(
            f"{prefix}_add", 
            0 if prefix != "out" else Z
        )
        mul_attr = helper.make_attribute(
            f"{prefix}_mul", 
            1 if prefix != "out" else M
        )
        levels_attr = helper.make_attribute(
            f"{prefix}_n_levels", 
            0 if prefix != "out" else 2**out_bit_width
        )
        shift_attr = helper.make_attribute(
            f"{prefix}_shift", 
            0 if prefix != "out" else 2 ** delta
        )
        
        node.attribute.extend([rq_attr, add_attr, mul_attr, levels_attr, shift_attr])
        
    node.attribute.append(
        helper.make_attribute(
            "add_bits",
            out_bit_width
        )
    )
    
    # remove quant node
    consumers = mw.find_consumers(quant_node.output[0])
    if consumers:
        for cons in consumers:
            for i in range(len(cons.input)):
                if cons.input[i] == quant_node.output[0]:
                    cons.input[i] = node.output[0]
    else:
        # no consumers -> this Quant feeds a graph output
        for out in mw.graph.output:
            if out.name == quant_node.output[0]:
                out.name = node.output[0]

    mw.graph.node.remove(quant_node)
        


class DoryResidualQuantParser(BaseTrasformation):
    def __init__(self, delta: int, verbose: bool = False):
        super().__init__(verbose)
        self.delta = delta

    def apply(self, mw: ModelWrapper):
        quant_nodes = [node for node in mw.graph.node if node.op_type == "Add"]
        
        # Iterate in REVERSE to avoid index/dependency issues
        for node in reversed(quant_nodes):
            merge_add_quant(mw, node, self.delta)
            
        return mw, False