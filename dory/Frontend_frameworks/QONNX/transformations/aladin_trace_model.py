import os
import csv
import numpy as np
from onnx import helper, TensorProto, NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from typing import *


MAX_PASS = 3
    
    
def find_upstream_quant(model: ModelWrapper, tensor_name: str) -> str:
    curr_name = tensor_name
    for _ in range(MAX_PASS):
        node = model.find_producer(curr_name)
        if node.op_type == "Quant":
            return node.input[3]
        if node is None:
            return
        
        curr_name = node.input[0]

    raise ValueError("Quant node not found!")    


def find_downstream_quant(model: ModelWrapper, tensor_name: str) -> str:
    curr_name = tensor_name
    for _ in range(MAX_PASS):
        node = model.find_consumer(curr_name)
        if node.op_type == "Quant":
            return node.input[3]
        
        curr_name = node.output[0]

    raise ValueError("Quant node not found!") 


def profile_convolution(model: ModelWrapper, node: NodeProto) -> Tuple[int, int, int]:
    """
    Profile a convolution operation using im2col transformation.

    Returns:
        macs: total number of multiply-accumulate operations
        total_mem: total memory footprint in bytes
    """
    input_shape = model.get_tensor_shape(node.input[0])
    output_shape = model.get_tensor_shape(node.output[0])
    weight_shape = model.get_tensor_shape(node.input[1])
    padding = tuple(get_by_name(node.attribute, "pads").ints)
    stride = tuple(get_by_name(node.attribute, "strides").ints)
    impl_attr = get_by_name(node.attribute, "implementation")
    if impl_attr is not None:
        implementation = impl_attr.s.decode("utf-8")
    else:
        implementation = "default"
    input_bitwidth = int(model.get_initializer(find_upstream_quant(model, node.input[0])))
    weight_bitwidth = int(model.get_initializer(find_upstream_quant(model, node.input[1])))
    output_bitwidth = int(model.get_initializer(find_downstream_quant(model, node.output[0])))
    _, C_in, H_in, W_in = input_shape
    C_out, _, K_h, K_w = weight_shape
    p_t, p_b, p_l, p_r = padding
    s1, s2 = stride

    H_out = (H_in + p_t + p_b - K_h) // s1 + 1
    W_out = (W_in + p_l + p_r - K_w) // s2 + 1

    # im2col matrix dimensions: [C_in * K_h * K_w, H_out * W_out]
    im2col_elems = C_in * K_h * K_w * H_out * W_out
    im2col_mem = im2col_elems * input_bitwidth // 8

    # standard buffers
    input_mem = np.prod(input_shape) * input_bitwidth // 8
    weight_mem = np.prod(weight_shape) * weight_bitwidth // 8
    # bias_mem = C_out * output_bitwidth // 8
    out_mem = np.prod(output_shape) * output_bitwidth // 8

    total_mem = input_mem + weight_mem + out_mem + im2col_mem
    if len(node.input) == 3:
        bias_shape = model.get_tensor_shape(node.input[2])
        bias_mem = np.prod(bias_shape) * 32 // 8
        total_mem += bias_mem
    
    macs = C_out * H_out * W_out * C_in * K_h * K_w
    bops = macs * (1 + input_bitwidth + weight_bitwidth + output_bitwidth)
    # LUT-based implementation adds additional lookup table storage
    if implementation == "lut":
        lut_dim = 2 ** (input_bitwidth + weight_bitwidth) * output_bitwidth // 8
        total_mem += lut_dim
        macs = 0

    return macs, total_mem, bops
    
    
def profile_gemm(model: ModelWrapper, node: NodeProto) -> Tuple[int, int, int]:
    input_shape = model.get_tensor_shape(node.input[0])
    output_shape = model.get_tensor_shape(node.output[0])
    weight_shape = model.get_tensor_shape(node.input[1])
    implementation = get_by_name(node.attribute, "implementation").s.decode("utf-8")
    input_bitwidth = int(model.get_initializer(find_upstream_quant(model, node.input[0])))
    weight_bitwidth = int(model.get_initializer(find_upstream_quant(model, node.input[1])))
    output_bitwidth = int(model.get_initializer(find_downstream_quant(model, node.output[0])))
    input_mem = np.prod(input_shape) * input_bitwidth // 8
    weight_mem = np.prod(weight_shape) * weight_bitwidth // 8
    out_mem = np.prod(output_shape) * output_bitwidth // 8
    
    total_mem = input_mem + weight_mem + out_mem
    if len(node.input) == 3:
        bias_shape = model.get_tensor_shape(node.input[2])
        bias_mem = np.prod(bias_shape) * output_bitwidth // 8
        total_mem += bias_mem
        
    macs = weight_shape[0] * weight_shape[1]
    bops = macs * (1 + input_bitwidth + output_bitwidth + weight_bitwidth)
    if implementation == "lut":
        lut_dim = 2 ** (input_bitwidth + weight_bitwidth) * output_bitwidth // 8
        total_mem += lut_dim
        macs = 0
    
    return macs, total_mem, bops
    

def profile_relu(model: ModelWrapper, node: NodeProto) -> Tuple[int, int, int]:
    input_shape = model.get_tensor_shape(node.input[0])
    input_bitwidth = int(model.get_initializer(find_upstream_quant(model, node.input[0])))
    input_mem = output_mem = np.prod(input_shape) * input_bitwidth
    total_mem = input_mem + output_mem
    macs = 0
    bops = np.prod(input_shape) * (input_bitwidth + 1)
    
    return macs, total_mem, bops 


def profile_quant(model: ModelWrapper, node: NodeProto) -> Tuple[int, int, int]:
    input_shape = model.get_tensor_shape(node.input[0])
    try:
        input_bitwidth = int(model.get_initializer(find_upstream_quant(model, node.input[0])))
    except:
        input_bitwidth = 8
    output_bitwidth = int(model.get_initializer(node.input[3]))
    impl_attr = get_by_name(node.attribute, "implementation")
    if impl_attr is not None:
        implementation = impl_attr.s.decode("utf-8")
    else:
        implementation = "default"
    scale = np.size(model.get_initializer(node.input[1]))
    
    input_mem = np.prod(input_shape) * input_bitwidth
    output_mem = np.prod(input_shape) * output_bitwidth
    
    if implementation == "thresholds":
        num_bins = (2 ** output_bitwidth) - 1
        bops = (np.log2(num_bins) * input_bitwidth) * np.prod(input_shape)
        param_mem = num_bins * input_bitwidth * scale
    else:
        bops = (2 * input_bitwidth + 5) + np.prod(input_shape)
        param_mem = scale * 32
        
    total_mem = input_mem + output_mem + param_mem
    
    return 0, total_mem, bops


def profile_avgpool(model: ModelWrapper, node: NodeProto) -> Tuple[int, int, int]:
    input_shape = model.get_tensor_shape(node.input[0])
    output_shape = model.get_tensor_shape(node.output[0])
    input_bitwidth = int(model.get_initializer(find_upstream_quant(model, node.input[0])))
    _, _, H_in, W_in = input_shape
    _, _, H_out, W_out = output_shape
    
    K_h = H_in // H_out
    K_w = W_in // W_out
    
    input_mem = output_mem = np.prod(input_shape) * input_bitwidth
    total_mem = input_mem + output_mem

    bops = np.prod(input_shape) * K_h * K_w * input_bitwidth
    
    return 0, total_mem, bops
    

def profile_maxpool(model: ModelWrapper, node: NodeProto) -> Tuple[int, int, int]:
    input_shape = model.get_tensor_shape(node.input[0])
    output_shape = model.get_tensor_shape(node.output[0])
    input_bitwidth = int(model.get_initializer(find_upstream_quant(model, node.input[0])))
    _, _, H_in, W_in = input_shape
    _, _, H_out, W_out = output_shape
    
    K_h = H_in // H_out
    K_w = W_in // W_out
    
    input_mem = output_mem = np.prod(input_shape) * input_bitwidth
    total_mem = input_mem + output_mem
    
    bops = np.prod(input_shape) * K_h * K_w * input_bitwidth
    
    return 0, total_mem, bops


def profile_add(model: ModelWrapper, node: NodeProto) -> Tuple[int, int, int]:
    """
    Profile an element-wise Add operation, including ONNX broadcasting.

    Returns:
        macs: Always zero because Add performs no multiply-accumulate operations.
        total_mem: Memory footprint of both inputs and the output, in bits.
        bops: Bit operations required for all element-wise additions.
    """
    input_a_shape = model.get_tensor_shape(node.input[0])
    input_b_shape = model.get_tensor_shape(node.input[1])
    output_shape = model.get_tensor_shape(node.output[0])

    input_a_bitwidth = int(
        model.get_initializer(
            find_upstream_quant(model, node.input[0])
        )
    )
    input_b_bitwidth = int(
        model.get_initializer(
            find_upstream_quant(model, node.input[1])
        )
    )
    output_bitwidth = int(
        model.get_initializer(
            find_downstream_quant(model, node.output[0])
        )
    )

    input_a_mem = np.prod(input_a_shape) * input_a_bitwidth
    input_b_mem = np.prod(input_b_shape) * input_b_bitwidth
    output_mem = np.prod(output_shape) * output_bitwidth

    total_mem = input_a_mem + input_b_mem + output_mem

    bits_per_add = max(input_a_bitwidth, input_b_bitwidth) + 1
    bops = np.prod(output_shape) * bits_per_add

    return 0, int(total_mem), int(bops)



class ImplementationAwareTrace(BaseTrasformation):

    def __init__(self, output_path: str, file_name: str, verbose: bool = False):
        self.output_path = output_path
        self.file_name = file_name
        super().__init__(verbose)
        

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        csv_path = os.path.join(self.output_path, f"{self.file_name}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        op_idx = 0
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Op.", "MACs", "Memory", "BOPs"])
            
            for node in graph.node:
                if node.op_type == "Conv":
                    macs, memory, bops = profile_convolution(model, node)
                elif node.op_type in ["Gemm", "Matmul"]:
                    macs, memory, bops = profile_gemm(model, node)
                elif node.op_type in ["Trunc", "Quant"]:
                    macs, memory, bops = profile_quant(model, node)
                elif node.op_type == "Relu":
                    macs, memory, bops = profile_relu(model, node)
                elif node.op_type in ["GlobalAveragePool", "AveragePool"]:
                    macs, memory, bops = profile_avgpool(model, node)
                elif node.op_type in ["GlobalMaxPool", "MaxPool"]:
                    macs, memory, bops = profile_maxpool(model, node)
                elif node.op_type == "Add":
                    macs, memory, bops = profile_add(model, node)
                else:
                    continue
                
                layer_name = f"{node.op_type}_{op_idx}"
                op_idx += 1
                layer_name = layer_name.replace("GlobalAveragePool", "avgpool")
                writer.writerow([layer_name, macs, memory, bops])
            
        return model, False