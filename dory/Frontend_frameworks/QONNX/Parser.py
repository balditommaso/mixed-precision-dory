# Libraries
import os
import numpy as np
import onnx as on
import onnxruntime as ort
from typing import *
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.exec_qonnx import exec_qonnx
from copy import deepcopy

# Trasformations
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.change_batchsize import ChangeBatchSize
from qonnx.transformation.infer_shapes import InferShapes
from dory.Frontend_frameworks.QONNX.transformations.dory_config_generator import DoryConfigParser
from dory.Frontend_frameworks.QONNX.transformations.fold_static_quant import FoldStaticQuant, min_int, max_int
from dory.Frontend_frameworks.QONNX.transformations.record_out_scale import RecordOutScale
from dory.Frontend_frameworks.QONNX.transformations.dory_relu_quant_parser import DoryActQuantParser
from dory.Frontend_frameworks.QONNX.transformations.dory_avg_pool_parser import DoryAvgPoolQuantParser
from dory.Frontend_frameworks.QONNX.transformations.dory_flatten_parser import DoryFlattenParser
from dory.Frontend_frameworks.QONNX.transformations.rename_tensors import RenameTensorsSequentially

# DORY modules
from dory.Frontend_frameworks.Quantlab.Parser import onnx_manager as Quantlab_onnx_manager


class onnx_manager(Quantlab_onnx_manager):
    
    def __init__(
        self, 
        onnx: str, 
        config_file: Dict[str, Any], 
        net_prefix: str = "", 
        log: str = "./logs/Frontend",
        verbose: bool = False
    ):
        # TODO: adapter QONNX -> Quantlab ONNX
        # setup log directory
        self.log_dir = os.path.join(log, "onnx_files/")
        os.system(f"rm -rf {self.log_dir}")
        os.system(f"mkdir -p {self.log_dir}")
        # load the model
        model = ModelWrapper(on.load(onnx))
        # apply transformations
        model = cleanup_model(model, override_inpsize=1)
        model.save(os.path.join(self.log_dir, "A_QONNX_cleanup.onnx"))
        qonnx_model = deepcopy(model)
        # fold static quantization
        model = model.transform(RecordOutScale(verbose=verbose))
        model = model.transform(FoldStaticQuant(verbose=verbose))
        model.save(os.path.join(self.log_dir, "B_QONNX_fold_static_quant.onnx"))
        # generate config.json file
        model = model.transform(DoryConfigParser(config=config_file, code_size=150000, verbose=verbose))
        model.save(os.path.join(self.log_dir, "C_QONNX_remove_input_quant.onnx"))
        # adapt to dory activation quantization
        transformed_onnx_path = os.path.join(self.log_dir, "D_QONNX_parse_quant_act.onnx")
        model = model.transform(DoryActQuantParser(delta=2**19, verbose=verbose))
        model = model.transform(DoryAvgPoolQuantParser(delta=2**19, verbose=verbose))
        model = model.transform(InferShapes())
        model = model.transform(DoryFlattenParser(verbose=verbose))
        model = model.transform(RenameTensorsSequentially(verbose=verbose))
        model.save(transformed_onnx_path)

        # TODO: check the correctness of the transformations
        self.check_flow(qonnx_model, transformed_onnx_path, config_file)
        
        # call the Quantlab manager
        super().__init__(transformed_onnx_path, config_file, net_prefix)


    def check_flow(
        self,
        qonnx_model: ModelWrapper, 
        transformed_model: str, 
        input_config: Dict[str, Any]
    ) -> None:
        # remove the input quantization from the QONNX layer

        qonnx_model_path = os.path.join(self.log_dir, "original.onnx")
        qonnx_model = qonnx_model.transform(DoryConfigParser(config={}))
        qonnx_model.save(qonnx_model_path)
        
        input_tensor_path = os.path.join(self.log_dir, "input.npy")
        input_bit = input_config["input_bits"]
        signed = input_config["input_signed"]
        in_shape = qonnx_model.get_tensor_shape("global_in")
        lb = min_int(signed, False, input_bit)
        ub = max_int(signed, False, input_bit)
        random_tensor = np.random.randint(lb, ub, size=in_shape).astype(np.float32)
        np.save(input_tensor_path, random_tensor)
        
        # run the tests
        exec_qonnx(qonnx_model_path, input_tensor_path, output_prefix=self.log_dir)
        qonnx_output = np.load(os.path.join(self.log_dir, "global_out_batch0.npy"))
        
        model = on.load(transformed_model)
        remove_names = {"out_scale", "weight_bits", "bias_bits", "input_bits", "out_bits", "min", "max"}
        for node in model.graph.node:
            keep_attrs = [a for a in node.attribute if a.name not in remove_names]
            if len(keep_attrs) != len(node.attribute):
                # clear all attributes then copy back the ones to keep
                node.ClearField("attribute")
                for a in keep_attrs:
                    node.attribute.add().CopyFrom(a)
        
        sess = ort.InferenceSession(model.SerializeToString())
        res =sess.run([sess.get_outputs()[0].name], {"0": random_tensor})[0]

        diff = qonnx_output - res

        print("Shape:", diff.shape)
        print("Mean absolute difference:", np.mean(np.abs(diff)))
        print("Max absolute difference:", np.max(np.abs(diff)))
        print("L2 norm difference:", np.linalg.norm(diff))
        print("Relative L2 error:", np.linalg.norm(diff) / np.linalg.norm(qonnx_output))
        print("Are they exactly equal?", np.allclose(qonnx_output, res))
        print("Are prediction equal?", qonnx_output.argmax() == res.argmax())
        return
