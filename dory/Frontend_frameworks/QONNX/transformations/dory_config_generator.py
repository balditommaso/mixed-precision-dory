from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from qonnx.util.basic import get_by_name
from typing import *



class DoryConfigParser(BaseTrasformation):
    """
    Generate the config file espected by DORY for the parsing
    and edit it based on the QONNX model informations
    
    TODO: handle 2-bit quantization
    """
    
    def __init__(self, config: Dict[str, Any], code_size: int = 150000, verbose: bool = False):
        super().__init__(verbose)
        self.config = config
        self.code_size = code_size
    
    
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        # defaults
        self.config["BNRelu_bits"] = 32
        self.config["code reserved space"] = self.code_size
        
        # we handle only models with just oe input
        self.config["n_inputs"] = 1
        input_quant = model.find_consumer("global_in")
        if input_quant.op_type != "Quant":  
            self.error_message(f"Missing input quantization!.", ValueError)
            
        self.config["input_bits"] = int(model.get_initializer(input_quant.input[3]))
        self.config["input_signed"] = bool(get_by_name(input_quant.attribute, "signed").i)
        
        # remove the input quantization node
        in_tensor = input_quant.input[0]
        out_tensor = input_quant.output[0]
        
        target_node = model.find_consumer(out_tensor)
        if target_node is None:
            self.error_message(f"Consumer of Quant_0 output not found, check the QONNX graph.")
        target_node.input[0] = in_tensor
        
        graph.node.remove(input_quant)
        
        return (model, False)
            
            
            
            
            