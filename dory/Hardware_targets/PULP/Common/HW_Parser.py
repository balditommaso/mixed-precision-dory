

# Libraries
import numpy as np
import json
import os

# DORY modules
from dory.Parsers import HW_node, Layer_node
from dory.Parsers.Parser_DORY_to_HW import Parser_DORY_to_HW
from functools import partial



class onnx_manager_PULP(Parser_DORY_to_HW):
    
    LAYERS_SUPPORTED = [
        "Convolution", 
        "Pooling", 
        "FullyConnected", 
        "Addition", 
        "QAddition",
        "ReluConvolution", 
        "ReluPooling", 
        "ReluFullyConnected", 
        "ReluAddition", 
        "ReluQAddition",
        "BNReluConvolution", 
        "RequantPooling", 
        "BNReluFullyConnected", 
        "BNReluAddition", 
        "BNReluQAddition"
    ]

    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(
        self, 
        graph: list[HW_node.HW_node], 
        config_file: dict = None, 
        config_file_dir: str = None, 
        n_inputs: int = 1, 
        verify_checksum: bool = True,
        num_cores: int = 8,
        L1_capacity: int = None,
        L2_capacity: int = None
    ):
        file_path = self.get_file_path()
        pattern_rewriter = self.get_pattern_rewriter()
        with open(os.path.join(file_path, "pattern_rules.json")) as f:
            rules = json.load(f)
        with open(os.path.join(file_path, "HW_description.json")) as f:
            HW_description = json.load(f)

        if L1_capacity is not None:
            HW_description["memory"]["L1"]["dimension"] = L1_capacity
            
        if L2_capacity is not None:
            HW_description["memory"]["L2"]["dimension"] = L2_capacity
        
        assert 1 <= num_cores <= 8
        
        try:
            db = HW_description['double_buffering']
        except KeyError:
            print("onnx_manager_PULP: Key 'double_buffering' not found in HW_description.json - setting to 2")
            db = 2

        self.double_buffering = db
        
            
        tiler = self.get_tiler()

        tiler = partial(tiler, double_buffering=self.double_buffering)
        
        network_directory = None
        if config_file_dir is not None and hasattr(config_file, "onnx_file"):
            network_directory = os.path.join(
                config_file_dir, 
                os.path.dirname(config_file["onnx_file"])
            )
            
        super().__init__(
            graph, 
            rules, 
            pattern_rewriter, 
            self.LAYERS_SUPPORTED, 
            HW_description,
            config_file, 
            tiler, 
            network_directory, 
            n_inputs, 
            verify_checksum,
            num_cores
        )

    def get_file_path(self):
        raise NotImplementedError("To be implemented by child class!")

    def get_pattern_rewriter(self):
        raise NotImplementedError("To be implemented by child class!")

    def get_tiler(self):
        raise NotImplementedError("To be implemented by child class!")

    def _get_weights_attr(self, node):
        weights_name = None
        for name in node.constant_names:
            if name not in ["l","k","outshift","outmul"]:
                if "bias" not in name:
                    weights_name = name
        assert weights_name is not None, f"Node  {node.name} of op {node.op_type} doesn't have weights."
        return getattr(node, weights_name)

    def adjust_node_data_layout(self, node, node_id):
        if "FullyConnected" in node.name:
            weights = self._get_weights_attr(node)
            if weights["layout"] == "CinCout":
                weights["value"] = weights["value"].T
                weights["layout"] = "CoutCin"
            prev_node = self.DORY_Graph[node_id-1]
            if node_id != 0 and prev_node.layout == "CHW":
                temp = weights["value"]
                temp = temp.reshape(node.output_channels, prev_node.output_channels, prev_node.output_dimensions[0], prev_node.output_dimensions[1])
                temp = np.transpose(temp, (0, 2, 3, 1))
                temp = temp.flatten()
                weights["value"] = temp
                # needed to compute final checksum for <8b layers
        elif "Convolution" in node.name:
            weights = self._get_weights_attr(node)
            if weights["layout"] == "CoutCinK":
                if node.conv1d:
                    weights["value"] = weights["value"][:,:,None,:]
                weights["value"] = np.transpose(weights["value"], (0,2,3,1))
                weights["layout"] = "CoutKCin"

    def adjust_data_layout(self):
        print("\nPULP Backend: Adjusting Data Layout to HWC and CoutKCin.")
        for i, node in enumerate(self.DORY_Graph):
             self.adjust_node_data_layout(node, i)

    def check_parameters(self):
        WARNINGS =0
        for node in self.DORY_Graph:
            for key, value in node.__dict__.items():
                if key not in HW_node.HW_node(Layer_node.Layer_node(), self.HW_description).__dict__.keys() and key not in Layer_node.Layer_node().__dict__.keys():
                    if key not in node.constant_names:
                        print("WARNING: DORY Backend. Attribute {} of Node {} is not inside the predefined parameters for DORY nodes.".format(key, node.name))
                        WARNINGS +=1
                if isinstance(value, list):
                    if len(value) == 0:
                        WARNINGS +=1
                        print("WARNING: DORY Backend. Attribute {} of Node {} is an empty list.".format(key, node.name))
                if isinstance(value, type(None)):
                    WARNINGS +=1
                    print("WARNING: DORY Backend. Attribute {} of Node {} is still not initialized.".format(key, node.name))
        print("\nDORY checking of the attribute of the graph: {} WARNINGS\n".format(WARNINGS))
