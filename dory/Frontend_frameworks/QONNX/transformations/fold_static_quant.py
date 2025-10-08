import numpy as np
from copy import deepcopy
from onnx import helper, numpy_helper
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from qonnx.util.basic import get_by_name
from typing import *


def is_static_quant(model: ModelWrapper, tensor_name: str) -> bool:
    if tensor_name == "global_in":
        return False
    
    return model.find_producer(tensor_name) is None


def min_int(signed: bool, narrow_range: bool, bit_width: int) -> int:
    """
    Compute the minimum integer representable by a given number of bits.
    """
    if signed and narrow_range:
        value = -(2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        value = -(2 ** (bit_width - 1))
    else:
        value = 0 * bit_width
    return value


def max_int(signed: bool, narrow_range: bool, bit_width: int) -> int:
    """
    Compute the maximum integer representable by a given number of bits.
    """
    if not signed and not narrow_range:
        value = (2**bit_width) - 1
    elif not signed and narrow_range:
        value = (2**bit_width) - 2
    else:
        value = (2 ** (bit_width - 1)) - 1
    return value


def int_quant(
    inp_tensor: np.ndarray, 
    scale: np.ndarray, 
    zeropt: np.ndarray, 
    bitwidth: int, 
    signed: bool, 
    narrow: bool, 
    rounding_mode: str
) -> np.ndarray:
    # Scaling
    y_int = inp_tensor / scale
    y_int = y_int + zeropt
    # clamping
    min_int_val = min_int(signed, narrow, bitwidth)
    max_int_val = max_int(signed, narrow, bitwidth)
    y_int = np.where(y_int > max_int_val, max_int_val, y_int)
    y_int = np.where(y_int < min_int_val, min_int_val, y_int)
    # rounding
    rounding_fx = resolve_rounding_mode(rounding_mode)
    y_int = rounding_fx(y_int)

    return y_int


def resolve_rounding_mode(mode_string: str) -> Callable:
    """
    Resolve the rounding mode string of IntQuant and Trunc ops
    to the corresponding numpy functions.
    """
    normalized_mode_string = mode_string.upper()
    if normalized_mode_string == "ROUND" or normalized_mode_string == "HALF_EVEN":
        return np.round
    elif normalized_mode_string == "CEIL":
        return np.ceil
    elif normalized_mode_string == "FLOOR":
        return np.floor
    elif normalized_mode_string == "UP":

        def round_up(x):
            return np.sign(x) * np.ceil(np.abs(x))

        return round_up
    elif normalized_mode_string == "DOWN":
        return np.fix
    elif normalized_mode_string == "HALF_UP":

        def round_half_up(x):
            return np.sign(x) * np.floor(np.abs(x) + 0.5)

        return round_half_up
    elif normalized_mode_string == "HALF_DOWN":

        def round_half_down(x):
            return np.sign(x) * np.ceil(np.abs(x) - 0.5)

        return round_half_down
    else:
        raise ValueError(f"Could not resolve rounding mode called: {normalized_mode_string}")



class FoldStaticQuant(BaseTrasformation):
    """
    Convert Quant node of the parameters, such as weight and bias, to 
    thir quantized repperesentation and redirect the new tensor as input 
    of the operation node and store useful information.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        
    
    
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        iter_graph = deepcopy(graph)
        for node in iter_graph.node:
            # check operation which could have static parameters
            if node.op_type != "Quant" or not is_static_quant(model, node.input[0]):
                continue
                        
            # apply trasformation
            target_node = model.find_consumer(node.output[0])
            if target_node is None:
                self.error_message(f"{node.name} is a dead leaf, check the QONNX graph.")
                
            param_shape = model.get_tensor_shape(node.input[0])
            is_bias = len(param_shape) == 1
            fp_value = model.get_initializer(node.input[0])
            scale = model.get_initializer(node.input[1])
            zeropt = model.get_initializer(node.input[2])
            bit_width = int(model.get_initializer(node.input[3]))
            
            is_narrow = bool(get_by_name(node.attribute, "narrow").i)
            rounding_mode = get_by_name(node.attribute, "rounding_mode").s.decode("utf-8")
            signed = bool(get_by_name(node.attribute, "signed").i)
            
            if bit_width not in [2, 4, 8, 16, 32]:
                self.error_message(f"Not supported bit_width for {node.name} ({bit_width}).", ValueError)
                
            q_value = int_quant(fp_value, scale, zeropt, bit_width, signed, is_narrow, rounding_mode).astype(np.float32)
            
            # needed by the backend to separate bias from weights
            quant_tensor_name = model.make_new_valueinfo_name()
            if is_bias:
                quant_tensor_name += "_bias"
                
            quant_tensor = numpy_helper.from_array(q_value, quant_tensor_name)
            
            # add tensor to the graph
            graph.initializer.append(quant_tensor)
            
            idx = list(target_node.input).index(node.output[0])
            target_node.input[idx] = quant_tensor.name
            
            # add bit_width information to the op. node
            attr = helper.make_attribute(
                "bias_bits" if is_bias else "weight_bits", 
                bit_width
            )
            target_node.attribute.append(attr)
            
            # remove old initzializers
            for name in node.input[:4]:  # fp_value, scale, zeropt, bit_width
                for init in list(graph.initializer):
                    if init.name == name:
                        graph.initializer.remove(init)
                        break
            
            # remove quant node
            graph.node.remove(node)
        
        return (model, False)
            
            
            
            
            
            
            
            