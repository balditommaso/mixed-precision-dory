from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper


AVAILABLE_OP = [
    "Conv",
    "Gemm",
    "MatMul",
]


class FoldStaticQuant(Transformation):
    """
    Convert Quant node of the parameters, such as weight and bias, to 
    thir quantized repperesentation and redirect the new tensor as input 
    of the operation node and store useful information.

    Args:
        Transformation (_type_): _description_
    """
    def apply(self, model: ModelWrapper):
        graph = model.graph
        for node in graph.node:
            # check operation which could have static parameters
            if node.op_type not in AVAILABLE_OP:
                continue
            
            