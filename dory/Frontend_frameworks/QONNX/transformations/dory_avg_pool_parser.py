import numpy as np
from copy import deepcopy
from onnx import helper, numpy_helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from dory.Frontend_frameworks.QONNX.transformations.fold_static_quant import *
from qonnx.util.basic import get_by_name
from typing import *



class DoryAvgPoolQuantParser(BaseTrasformation):
    """
    Adapter from QONNX average pool via truncate to DORY average pool harmonization.
    """
    
    def __init__(self, delta: int, verbose: bool = False):
        super().__init__(verbose)
        self.delta = delta
        
    
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        iter_graph = deepcopy(graph)
        for node in iter_graph.node:
            if node.op_type != "Trunc":
                continue
            
            avg_node = model.find_producer(node.input[0])
            if avg_node is None or avg_node.op_type not in ["GlobalAveragePool", "AveragePool"]:
                self.warning_message(f"{node.name} does not follow an AveragePool node.")
                continue
            
            # inputs
            in_scale = model.get_initializer(node.input[1])
            zeropt = model.get_initializer(node.input[2])
            in_bit_width = int(model.get_initializer(node.input[3]))
            out_scale = model.get_initializer(node.input[4])
            out_bit_width = int(model.get_initializer(node.input[5]))
            
            # attributes
            is_narrow = bool(get_by_name(node.attribute, "narrow").i)
            rounding_mode = get_by_name(node.attribute, "rounding_mode").s.decode("utf-8")
            signed = bool(get_by_name(node.attribute, "signed").i)
            
            round_fx = resolve_rounding_mode(rounding_mode)
            
            M = round_fx(out_scale / in_scale * self.delta).astype(np.float32)
            
            M = numpy_helper.from_array(M, model.make_new_valueinfo_name())
            graph.initializer.append(M)
            
            out_mul_tensor = helper.make_tensor_value_info(
                model.make_new_valueinfo_name(),
                TensorProto.FLOAT,
                model.get_tensor_shape(node.input[0])
            )
            graph.value_info.append(out_mul_tensor)
            
            mul_node = helper.make_node(
                "Mul",
                [avg_node.output[0], M.name],
                [out_mul_tensor.name]
            )
            graph.node.append(mul_node)
            div_input_name = out_mul_tensor.name
            
            # add node on for asymmetric qunatization
            if zeropt is not None and not np.all(zeropt == 0):
                Z = np.round(zeropt * self.delta * out_scale)
                Z = numpy_helper.from_array(Z, model.make_new_valueinfo_name())
                graph.initializer.append(Z)
                
                out_add_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    model.get_tensor_shape(div_input_name)
                )
                graph.value_info.append(out_add_tensor)
                
                add_node = helper.make_node(
                    "Add",
                    [out_mul_tensor.name, Z.name],
                    [out_add_tensor.name]
                )
                graph.node.append(add_node)
                div_input_name = out_add_tensor.name
                
            # div node to remove the scale
            D = np.array(self.delta, dtype=np.float32)
            D = numpy_helper.from_array(D, model.make_new_valueinfo_name())
            graph.initializer.append(D)
                        
            out_div_tensor = helper.make_tensor_value_info(
                model.make_new_valueinfo_name(),
                TensorProto.FLOAT,
                model.get_tensor_shape(div_input_name)
            )
            graph.value_info.append(out_div_tensor)
            
            div_node = helper.make_node(
                "Div",
                [div_input_name, D.name],
                [out_div_tensor.name]
            )
            graph.node.append(div_node)
            
            # clip to apply the relu activation
            out_clip_tensor = helper.make_tensor_value_info(
                model.make_new_valueinfo_name(),
                TensorProto.FLOAT,
                model.get_tensor_shape(out_div_tensor.name)
            )
            
            clip_node = helper.make_node(
                "Clip",
                [out_div_tensor.name],
                [out_clip_tensor.name]
            )
            
            clip_min = min_int(signed, is_narrow, out_bit_width)
            clip_max = max_int(signed, is_narrow, out_bit_width)
            
            bit_width_attr = helper.make_attribute(
                "out_bits", out_bit_width
            )
            clip_min_attr = helper.make_attribute(
                "min", clip_min
            )
            clip_max_attr = helper.make_attribute(
                "max", clip_max
            )
            clip_node.attribute.extend([bit_width_attr, clip_min_attr, clip_max_attr])
            graph.node.append(clip_node)
            # reconnecte the chain
            next_node = model.find_consumer(node.output[0])
            next_node.input[0] = out_clip_tensor.name
            
            graph.node.remove(node)
                        
            
        return (model, False)
            
            