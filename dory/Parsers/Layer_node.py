
import os
import numpy as np
import onnx
from .DORY_node import DORY_node


class Layer_node(DORY_node):

    def __init__(self):
        super().__init__()
        self.kernel_shape = [] # fH x fW
        self.dilations = [] 
        self.group = None
        self.strides = [] 
        self.input_channels = None 
        self.output_channels = None
        self.input_dimensions = [] # H x W
        self.output_dimensions = [] # H x W
        self.pads    = [] # Top, Left, Bottom, Right
        self.MACs = None
        self.weight_memory = None
        self.bias_memory = None
        self.constants_memory = None
        self.input_activation_memory = None
        self.output_activation_memory = None
        self.layout = None
        self.prefix = ""


    def update_input_dimensions(self, activation_tensor: onnx.TensorProto, Layer_parameters: dict) -> dict:
        if activation_tensor.name in self.input_indexes:
            dimension = activation_tensor.type.tensor_type.shape.dim
            try:
                Layer_parameters["input_channels"] = dimension[1].dim_value
                dims = [e.dim_value for e in dimension[2:]]
                if len(dims) == 1:
                    dims =  [1] + dims
                Layer_parameters["input_dimensions"] = dims
            except IndexError:
                Layer_parameters["input_channels"] = None
                Layer_parameters["input_dimensions"] = []

            if len(Layer_parameters["input_dimensions"]) == 0:
                Layer_parameters["input_dimensions"] =  [1, 1]
        
        return Layer_parameters


    def update_output_dimensions(self, activation_tensor: onnx.TensorProto, Layer_parameters: dict) -> dict:
        if activation_tensor.name == self.output_index:
            dimension = activation_tensor.type.tensor_type.shape.dim
            Layer_parameters["output_channels"] = dimension[1].dim_value
            dims = [e.dim_value for e in dimension[2:]]
            
            if len(dims) == 1:
                dims = [1] + dims
                
            Layer_parameters["output_dimensions"] = dims
            if len(Layer_parameters["output_dimensions"]) == 0:
                Layer_parameters["output_dimensions"] =  [1, 1]
                
        return Layer_parameters


    def populate_Layer_node(self, node_iterating: DORY_node, graph: list[DORY_node], prefix: str = "") -> None:
        self.populate_DORY_node(node_iterating, graph)
        Layer_parameters = {}
        Layer_parameters['prefix'] = prefix
        
        if self.name in ['FullyConnected', 'Addition']:
            Layer_parameters['kernel_shape'] = [1, 1]
            Layer_parameters['dilations'] = [1, 1]
            Layer_parameters['strides'] = [1, 1]
            Layer_parameters['group'] = 1
            
        if self.name in ['Pooling']:
            Layer_parameters['dilations'] = [1, 1]
            Layer_parameters['group'] = 1
            
        Layer_parameters['pads'] = [0, 0, 0, 0]

        for attribute in node_iterating.attribute:
            if attribute.name in ['kernel_shape', 'dilations', 'group', 'strides', 'pads']:
                if bool(attribute.i):
                    Layer_parameters[attribute.name] = attribute.i
                    
                if bool(attribute.ints):
                    Layer_parameters[attribute.name] = list(attribute.ints)
        
        for activation_tensor in graph.graph.input:
            Layer_parameters = self.update_input_dimensions(activation_tensor, Layer_parameters)
        
        for activation_tensor in graph.graph.value_info:
            Layer_parameters = self.update_input_dimensions(activation_tensor, Layer_parameters)
            Layer_parameters = self.update_output_dimensions(activation_tensor, Layer_parameters)
        
        for activation_tensor in graph.graph.output:
            Layer_parameters = self.update_output_dimensions(activation_tensor, Layer_parameters)
        
        if 'Global' in node_iterating.op_type:
            Layer_parameters['kernel_shape'] = Layer_parameters['input_dimensions']
            Layer_parameters['strides'] = [1, 1]
        
        if 'group' in Layer_parameters and Layer_parameters['group'] > 1:
            if not (Layer_parameters['group'] == Layer_parameters["output_channels"] == Layer_parameters["input_channels"]):
                raise ValueError("Depthwise convolution with input channels != output channels != groups")
   
        self.add_existing_dict_parameter(Layer_parameters)


    def add_memory_and_MACs(self) -> None:
        if "Convolution" in self.name or "FullyConnected" in self.name:
            self.add_existing_parameter("MACs", int(
                    np.prod(self.output_dimensions) 
                    * self.output_channels 
                    * self.input_channels 
                    * np.prod(self.kernel_shape) 
                    / self.group
                )
            )
            if self.group == 1:
                self.add_existing_parameter("weight_memory", int(
                        self.output_channels 
                        * self.input_channels 
                        * np.prod(self.kernel_shape) 
                        / self.group 
                        * self.weight_bits 
                        / 8
                    )
                )
            else:
                self.add_existing_parameter("weight_memory", int(
                        self.output_channels 
                        * self.input_channels 
                        * np.prod(self.kernel_shape) 
                        / self.group 
                        * 16 
                        * self.weight_bits 
                        / 8
                    )
                )
        else:
            self.add_existing_parameter("MACs", int(0))
            self.add_existing_parameter("weight_memory", int(0))
            
        if self.implementation == "lut":
            self.add_existing_parameter("MACs", int(0))
            
        self.add_existing_parameter("input_activation_memory", int(
                np.prod(self.input_dimensions) 
                * self.input_channels 
                * self.input_activation_bits 
                / 8
            )
        )
        self.add_existing_parameter("output_activation_memory", int(
                np.prod(self.output_dimensions) 
                * self.output_channels 
                * self.output_activation_bits 
                / 8
            )
        )
        constants_memory = 0
        bias_memory = 0
        for name in self.constant_names:
            if name in ["l","k"]:
                constants_memory += self.output_channels * self.constant_bits / 8
            if "bias" in name:
                bias_memory += self.output_channels * self.bias_bits / 8
                
        if self.group == 1:
            self.add_existing_parameter("bias_memory", int(bias_memory))
        else:
            self.add_existing_parameter("bias_memory", int(bias_memory * 16))
        
        self.add_existing_parameter("constants_memory", int(constants_memory))


    def export_to_dict(self) -> dict:
        node_dict = {}
        node_dict["name"] = self.name
        node_dict["DORY_node_parameters"] = {}
        node_dict["Layer_node_parameters"] = {}
        node_dict["Weights"] = {}
        for key, value in self.__dict__.items():
            if (
                not isinstance(value, dict) 
                and key != "name" 
                and key in DORY_node().__dict__.keys()
            ):
                node_dict["DORY_node_parameters"][key] = value
            elif not isinstance(value, dict) and key != "name":
                node_dict["Layer_node_parameters"][key] = value
            elif key != "name":
                node_dict["Weights"][key] = {}
                node_dict["Weights"][key]["Present"] = 'Yes'
                node_dict["Weights"][key]["Layout"] = value["layout"]
                
        return node_dict
