# Libraries
import json
import os
import numpy as np
import onnx as on
from qonnx.core.modelwrapper import ModelWrapper

# Trasformations
from qonnx.util.cleanup import cleanup_model

# DORY modules
from dory.Frontend_frameworks.Quantlab.Parser import onnx_manager as Quantlab_onnx_manager


class onnx_manager(Quantlab_onnx_manager):
    
    def __init__(self, onnx: str, config_file: str, net_prefix: str = "", log: str = "./logs/Frontend"):
        # TODO: adapter QONNX -> Quantlab ONNX
        # setup log directory
        self.log_dir = os.path.join(log, "onnx_files")
        os.system(f"rm -rf {self.log_dir}")
        os.system(f"mkdir -p {self.log_dir}")
        # load the model
        model = ModelWrapper(on.load(onnx))
        # apply transformations
        model = cleanup_model(model)
        model.save(os.path.join(self.log_dir, "A_QONNX_cleanup.onnx"))
        # TODO: generate config.json file
        # call the Quantlab manager
        super().__init__(onnx, config_file, net_prefix)


