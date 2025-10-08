import numpy as np
from copy import deepcopy
from onnx import helper, numpy_helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from dory.Frontend_frameworks.QONNX.transformations.fold_static_quant import *
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
            if quant_node is None or quant_node.op_type != "Quant":
                self.warning_message(f"{node.name} is not followed by a Quant node.")
                continue
            
            prev_node = model.find_producer(node.input[0])
            if prev_node is None:
                self.error_message(f"{node.name} has no previos node.", ValueError)
            
            # Mul node with the scale function multiplied by the scale
            out_scale = numpy_helper.to_array(get_by_name(prev_node.attribute, "out_scale").t)
            quant_scale = model.get_initializer(quant_node.input[1])
            rounding_mode = get_by_name(quant_node.attribute, "rounding_mode").s.decode("utf-8")
            round_fx = resolve_rounding_mode(rounding_mode)
            
            out_shape = model.get_tensor_shape(prev_node.output[0])
            C_out = out_shape[1] if len(out_shape) > 1 else out_shape[0]
            
            M = round_fx(out_scale / quant_scale * self.delta)
            # NOTE: channel-wise not supported in Dory
            # if np.isscalar(M) or np.size(M) == 1:
            #     M = np.full((C_out, 1, 1), float(M), dtype=np.float32)
            # else:
            #     M = np.reshape(M, (C_out, 1, 1)).astype(np.float32)
            
            M = numpy_helper.from_array(M, model.make_new_valueinfo_name())
            graph.initializer.append(M)
            
            out_mul_tensor = helper.make_tensor_value_info(
                model.make_new_valueinfo_name(),
                TensorProto.FLOAT,
                model.get_tensor_shape(prev_node.output[0])
            )
            graph.value_info.append(out_mul_tensor)
            
            mul_node = helper.make_node(
                "Mul",
                [prev_node.output[0], M.name],
                [out_mul_tensor.name]
            )
            graph.node.append(mul_node)
            
            div_input_name = out_mul_tensor.name
            # add node only for asymmetric quantization
            zeropt = model.get_initializer(quant_node.input[2])
            if zeropt is not None and not np.all(zeropt == 0.):
                Z = round_fx(zeropt * self.delta * out_scale)
                if np.isscalar(Z) or np.size(Z) == 1:
                    Z = np.full((C_out, 1, 1), float(Z), dtype=np.float32)
                else:
                    Z = np.reshape(Z, (C_out, 1, 1)).astype(np.float32)
                    
                Z = numpy_helper.from_array(Z, model.make_new_valueinfo_name())
                graph.initializer.append(Z)
                
                out_add_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    model.get_tensor_shape(prev_node.output[0])
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
                model.get_tensor_shape(prev_node.output[0])
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
                model.get_tensor_shape(prev_node.output[0])
            )
            
            clip_node = helper.make_node(
                "Clip",
                [out_div_tensor.name],
                [out_clip_tensor.name]
            )
            
            out_bit_width = int(model.get_initializer(quant_node.input[3]))
            is_narrow = bool(get_by_name(quant_node.attribute, "narrow").i)
            signed = bool(get_by_name(quant_node.attribute, "signed").i)
            
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
            next_node = model.find_consumer(quant_node.output[0])
            next_node.input[0] = out_clip_tensor.name
            
            graph.node.remove(node)
            graph.node.remove(quant_node)
            
            
        return (model, False)
            
            
