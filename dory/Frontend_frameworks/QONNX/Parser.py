# Libraries
import os
import json
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
from dory.Frontend_frameworks.QONNX.transformations.dory_quant_parser import DoryQuantParser
from dory.Frontend_frameworks.QONNX.transformations.dory_residual_parser import DoryResidualQuantParser
from dory.Frontend_frameworks.QONNX.transformations.dory_avg_pool_parser import DoryAvgPoolQuantParser
from dory.Frontend_frameworks.QONNX.transformations.dory_flatten_parser import DoryFlattenParser
from dory.Frontend_frameworks.QONNX.transformations.rename_tensors import RenameTensorsSequentially
from dory.Frontend_frameworks.QONNX.transformations.record_implementation import RecordImplementation


# DORY modules
from dory.Frontend_frameworks.Quantlab.Parser import onnx_manager as Quantlab_onnx_manager


class onnx_manager(Quantlab_onnx_manager):
    
    def __init__(
        self, 
        onnx_path: str, 
        config_file: Dict[str, Any],
        net_prefix: str = "", 
        log: str = "./logs/Frontend",
        delta: int = 16,
        verbose: bool = False
    ):
        print("")
        print("###################################")
        print("## DORY GENERAL PARSING OF QONNX ##")
        print("## FINAL RAPRESENTATION: DORY IR ##")
        print("###################################")
        self.log_dir = os.path.join(log, "onnx_files/")
        self.model_dir = os.path.dirname(onnx_path)
        os.system(f"rm -rf {self.log_dir}")
        os.system(f"mkdir -p {self.log_dir}")
        model = ModelWrapper(on.load(onnx_path))
        model = cleanup_model(model, override_inpsize=1)
        model.save(os.path.join(self.log_dir, "A_QONNX_cleanup.onnx"))
        model = model.transform(RecordOutScale(verbose=verbose))
        model = model.transform(FoldStaticQuant(verbose=verbose))
        model.save(os.path.join(self.log_dir, "B_QONNX_fold_static_quant.onnx"))
        model = model.transform(DoryConfigParser(config=config_file, code_size=150000, verbose=verbose))
        model.save(os.path.join(self.log_dir, "C_QONNX_remove_input_quant.onnx"))
        transformed_onnx_path = os.path.join(self.log_dir, "D_QONNX_parse_quant_act.onnx")
        model = model.transform(DoryResidualQuantParser(delta=delta))
        model = model.transform(DoryQuantParser(delta=delta, verbose=verbose))
        model = model.transform(DoryAvgPoolQuantParser(delta=delta, verbose=verbose))
        model = model.transform(InferShapes())
        model = model.transform(DoryFlattenParser(verbose=verbose))
        model = model.transform(RenameTensorsSequentially(verbose=verbose))

        model.save(transformed_onnx_path)
        print("QONNX conversion complete!\nValidation...")

        super().__init__(transformed_onnx_path, config_file, net_prefix)


    def clean_model(self, mw: ModelWrapper) -> ModelWrapper:
        graph = mw.graph

        extra_attr = ["out_scale", "weight_bits", "bias_bits", "input_bits", "out_bits", "implementation"]
        convert_to_inputs = ["min", "max"]

        for idx, node in enumerate(graph.node):
            keep_attrs = []
            remove_names = extra_attr
            for attr in node.attribute:
                if attr.name in convert_to_inputs:
                    tensor_name = f"{node.name}_{attr.name}_const_{idx}"
                    const_tensor = on.helper.make_tensor(
                        name=tensor_name,
                        data_type=on.TensorProto.FLOAT,
                        dims=[],
                        vals=[attr.i],
                    )
                    const_node = on.helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[tensor_name],
                        value=const_tensor
                    )
                    graph.node.insert(0, const_node)
                    node.input.append(tensor_name)
                elif attr.name not in remove_names:
                    keep_attrs.append(attr)

            if len(keep_attrs) != len(node.attribute):
                node.ClearField("attribute")
                for a in keep_attrs:
                    node.attribute.add().CopyFrom(a)


    def check_flow(
        self,
        qonnx_model: ModelWrapper, 
        transformed_model: str, 
        input_config: Dict[str, Any]
    ) -> None:
        qonnx_model_path = os.path.join(self.log_dir, "original.onnx")
        qonnx_model = qonnx_model.transform(DoryConfigParser(config={}))
        self.clean_model(qonnx_model)
        qonnx_model.save(qonnx_model_path)

        onnx_input = qonnx_model.get_metadata_prop("input")
        input_tensor_path = os.path.join(self.log_dir, "input.npy")
        if onnx_input is None:
            print("Testing with random inputs!")
            input_bit = input_config["input_bits"]
            signed = input_config["input_signed"]
            in_shape = qonnx_model.get_tensor_shape("global_in")
            lb = min_int(signed, False, input_bit)
            ub = max_int(signed, False, input_bit)
            input_tensor = np.random.randint(lb, ub, size=in_shape).astype(np.float32)
        else:
            input_tensor = np.array(json.loads(onnx_input))
        
        np.save(input_tensor_path, input_tensor)
        exec_qonnx(qonnx_model_path, input_tensor_path, output_prefix=self.log_dir)
        qonnx_output = np.load(os.path.join(self.log_dir, "global_out_batch0.npy"))
        
        model = on.load(transformed_model)
        self.clean_model(model)
        
        sess = ort.InferenceSession(model.SerializeToString())
        dory_output =sess.run([sess.get_outputs()[0].name], {"0": input_tensor})[0]    
        print("Are prediction equal?", qonnx_output.argmax() == dory_output.argmax())
        
        np.savetxt(
            os.path.join(self.model_dir, "input.txt"),
            input_tensor.reshape(-1, 1), 
            fmt="%d",
            delimiter=","
        )
        
        return
