
# Libraries
import json
import os
import numpy as np
import shutil

# DORY modules
from dory.Parsers.Parser_HW_to_C import Parser_HW_to_C
import dory.Utils.Templates_writer.Layer2D_template_writer as Layer2D_writer
import dory.Utils.Templates_writer.Makefile_template_writer as Makefile_writer
from dory.Utils.Templates_writer.TemplateWriter import TemplateWriter
import dory.Hardware_targets.PULP.Backend_Kernels.BackendKernelsAdapter as BackendKernelsAdapter


class C_Parser_PULP(Parser_HW_to_C):
    """ 
    Used to manage the ONNX files. 
    By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    """
    def __init__(
        self, 
        graph: list, 
        config_file: dict[str, any], 
        verbose_level: str, 
        perf_layer: str, 
        model_dir: str = None, 
        precision_library: str = "auto", 
        app_directory: str = "./application", 
        n_inputs: int = 1,
        L1_capacity: int = None,
        L2_capacity: int = None,
        num_cores: int = 8
    ):
        file_path = self.get_file_path()
        with open(os.path.join(file_path, "HW_description.json")) as f:
            HW_description = json.load(f)
            
        if L1_capacity is not None:
            HW_description["memory"]["L1"]["dimension"] = L1_capacity
            
        if L2_capacity is not None:
            HW_description["memory"]["L2"]["dimension"] = L2_capacity
            
        if precision_library == "auto":
            self.precision_library = C_Parser_PULP._auto_precision_library(graph) 
        else:
            self.precision_library = precision_library
            
        self.source_Constant_bits_library = config_file["BNRelu_bits"]
        self.config_file = config_file

        super().__init__(
            graph, 
            model_dir, 
            HW_description, 
            verbose_level, 
            perf_layer, 
            "Makefile", 
            app_directory, 
            n_inputs,
            num_cores
        )
        
        try:
            db = HW_description['double_buffering']
        except KeyError:
            print("C_Parser_PULP: Key 'double_buffering' not found in HW_description.json - setting to 2")

            db = 2
        self.double_buffering = db


    @staticmethod
    def _auto_precision_library(graph):
        precision_library = "8bit"
        for node in graph:
            if "Addition" not in node.name and "Pool" not in node.name:
                if node.get_parameter('output_activation_bits') < 8 or \
                    node.get_parameter('input_activation_bits') < 8 or \
                    node.get_parameter('weight_bits') < 8:
                    
                    precision_library = 'mixed-sw'
            else:
                if node.get_parameter('output_activation_bits') < 8 or \
                    node.get_parameter('input_activation_bits') < 8:
                        
                    precision_library = 'mixed-sw'
        return precision_library


    def node_backend_library(self, node):
        return self.precision_library


    def copy_backend_files(self, node, backend_library):
        if backend_library == "8bit":
            backendKernelsAdapter = BackendKernelsAdapter.PulpNNAdapter(
                "pulp-nn", 
                node, 
                self.source_Constant_bits_library
            )
        elif backend_library == "mixed-sw":
            backendKernelsAdapter = BackendKernelsAdapter.PulpMixedAdapter(
                "pulp-nn-mixed", 
                node, 
                self.source_Constant_bits_library, 
                "sw"
            )
        elif backend_library == "mixed-hw":
            backendKernelsAdapter = BackendKernelsAdapter.PulpMixedAdapter(
                "pulp-nn-mixed", 
                node, 
                self.source_Constant_bits_library, 
                "hw"
            )
        elif backend_library == "ne16":
            backendKernelsAdapter = BackendKernelsAdapter.PulpNNXAdapter(
                "pulp-nnx", 
                node, 
                "ne16"
            )
        else:
            raise ValueError(f"Unrecognised backend library: {backend_library}")

        for file in backendKernelsAdapter.get_src_files():
            shutil.copy(file, self.src_dir)

        for file in backendKernelsAdapter.get_inc_files():
            shutil.copy(file, self.inc_dir)


    def l2_template_keywords(self, node, backend_library):
        return Layer2D_writer.l2_layer_template(
            node, 
            backend_library, 
            double_buffering=self.double_buffering,
            num_cores=self.num_cores
        )


    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        n_memory_levels = self.HW_description['memory']['levels']

        for i, node in enumerate(self.HWgraph):
            backend_library = self.node_backend_library(node)
            self.copy_backend_files(node, backend_library)

            if n_memory_levels > 2 and \
                (
                    node.L3_input != 0 or \
                        (node.tiling_dimensions["L3"]["output_dimensions"] != node.tiling_dimensions["L2"]["output_dimensions"]) \
                        or \
                        (node.tiling_dimensions["L3"]["weights_dimensions"] != node.tiling_dimensions["L2"]["weights_dimensions"]) \
                ):
                tk = Layer2D_writer.l3_layer_template(node, self.num_cores)
                TemplateWriter.write(tk, {
                    os.path.join(self.src_dir, node.prefixed_name + ".c"): os.path.join(self.tmpl_dir, "layer_L3_c_template.c.t"),
                    os.path.join(self.inc_dir, node.prefixed_name + ".h"): os.path.join(self.tmpl_dir, "layer_L3_h_template.h.t")
                })
                
                if node.tiling_dimensions["L3"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
                    node.tiling_dimensions["L2"]["output_dimensions"][1] = int(
                        np.floor(
                            (node.tiling_dimensions["L2"]["input_dimensions"][1] \
                                - node.kernel_shape[0] \
                                + node.strides[0]) / node.strides[0]
                        )
                    )
                if node.tiling_dimensions["L3"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
                    # i_dim = o_dim * stride + k - stride
                    node.tiling_dimensions["L2"]["input_dimensions"][1] = node.tiling_dimensions["L2"]["output_dimensions"][1] \
                        * node.strides[0] \
                        + node.kernel_shape[0] \
                        - node.strides[0]
                        
                node.name = node.name + "_L2"
                padding = node.pads
                node.pads = [0, padding[1], 0, padding[3]]
                tk = self.l2_template_keywords(node, backend_library)
                TemplateWriter.write(tk, self.l2_template_mapping(node, backend_library))
                node.name = node.name[:-3]
                if padding[0] > 0:
                    node.name = node.name + "_L2_p_t"
                    node.pads = [padding[0], padding[1], 0, padding[3]]
                    tk = self.l2_template_keywords(node, backend_library)
                    TemplateWriter.write(tk, self.l2_template_mapping(node, backend_library))
                    node.name = node.name[:-1] + "b"
                    node.pads = [0, padding[1], padding[2], padding[3]]
                    # i_L2 -= (p_bot - ((i_L3 + p_top + p_bot) - (o_L3 * stride + k - stride)))
                    node.tiling_dimensions["L2"]["input_dimensions"][1] -= (padding[2] - \
                        (
                            (
                                node.tiling_dimensions["L3"]["input_dimensions"][1] \
                                + padding[0] \
                                + padding[2] \
                            )
                                - \
                            (
                                node.tiling_dimensions["L3"]["output_dimensions"][1] \
                                * node.strides[0] \
                                + node.kernel_shape[0] \
                                - node.strides[0] \
                            )
                        )
                    )
                    if node.tiling_dimensions["L1"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
                        node.tiling_dimensions["L1"]["input_dimensions"][1] = node.tiling_dimensions["L2"]["input_dimensions"][1]
                    if node.tiling_dimensions["L1"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
                        node.tiling_dimensions["L1"]["output_dimensions"][1] = node.tiling_dimensions["L2"]["output_dimensions"][1]
                    tk = self.l2_template_keywords(node, backend_library)
                    TemplateWriter.write(tk, self.l2_template_mapping(node, backend_library))
                    node.name = node.name[:-7]
            else:
                if node.tiling_dimensions["L2"]["input_dimensions"][2] == node.tiling_dimensions["L1"]["input_dimensions"][2]:
                    node.tiling_dimensions["L1"]["output_dimensions"][2] = int(
                        (
                            node.tiling_dimensions["L1"]["input_dimensions"][2] \
                            + (node.pads[1] + node.pads[3]) \
                            - node.kernel_shape[1] \
                            + node.strides[1] \
                        ) \
                        / node.strides[1]
                    )
                if node.tiling_dimensions["L2"]["input_dimensions"][1] == node.tiling_dimensions["L1"]["input_dimensions"][1]:
                    node.tiling_dimensions["L1"]["output_dimensions"][1] = int(
                        (
                            node.tiling_dimensions["L1"]["input_dimensions"][1] \
                            + (node.pads[0] + node.pads[2]) \
                            - node.kernel_shape[0] \
                            + node.strides[0]
                        ) \
                        / node.strides[0]
                    )
                tk = self.l2_template_keywords(node, backend_library)
                TemplateWriter.write(tk, self.l2_template_mapping(node, backend_library))

    def mapping_makefile(self):
        super(C_Parser_PULP, self).mapping_makefile()
        # also print the "vars.mk"
        prefix = self.HWgraph[0].prefix
        Makefile_writer.print_template_Makefile(
            self.HWgraph,
            self.HW_description,
            prefix+"vars.mk",
            self.app_directory,
            template_location_rel="Templates/vars.mk_template")
