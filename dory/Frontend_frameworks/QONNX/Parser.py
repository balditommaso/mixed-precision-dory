# Libraries
import json
import os
import numpy as np
import onnx as on
from qonnx.core.modelwrapper import ModelWrapper

# Trasformations
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.change_batchsize import ChangeBatchSize
from dory.Frontend_frameworks.QONNX.transformations.dory_config_generator import DoryConfigParser
from dory.Frontend_frameworks.QONNX.transformations.fold_static_quant import FoldStaticQuant
from dory.Frontend_frameworks.QONNX.transformations.record_out_scale import RecordOutScale
from dory.Frontend_frameworks.QONNX.transformations.dory_relu_quant_parser import DoryActQuantParser

# DORY modules
from dory.Frontend_frameworks.Quantlab.Parser import onnx_manager as Quantlab_onnx_manager


class onnx_manager(Quantlab_onnx_manager):
    
    def __init__(
        self, 
        onnx: str, 
        config_file: str, 
        net_prefix: str = "", 
        log: str = "./logs/Frontend",
        verbose: bool = False
    ):
        # TODO: adapter QONNX -> Quantlab ONNX
        # setup log directory
        self.log_dir = os.path.join(log, "onnx_files")
        os.system(f"rm -rf {self.log_dir}")
        os.system(f"mkdir -p {self.log_dir}")
        # load the model
        model = ModelWrapper(on.load(onnx))
        # apply transformations
        model = model.transform(ChangeBatchSize(1))
        model = cleanup_model(model)
        model.save(os.path.join(self.log_dir, "A_QONNX_cleanup.onnx"))
        # fold static quantization
        model = model.transform(RecordOutScale(verbose=verbose))
        model = model.transform(FoldStaticQuant(verbose=verbose))
        model.save(os.path.join(self.log_dir, "B_QONNX_fold_static_quant.onnx"))
        # generate config.json file
        model = model.transform(DoryConfigParser(config=config_file, code_size=150000, verbose=verbose))
        model.save(os.path.join(self.log_dir, "C_QONNX_remove_input_quant.onnx"))
        # adapt to dory activation quantization
        model = model.transform(DoryActQuantParser(delta=2**19, verbose=verbose))
        model.save(os.path.join(self.log_dir, "D_QONNX_parse_quant_act.onnx"))

        # call the Quantlab manager
        super().__init__(onnx, config_file, net_prefix)


