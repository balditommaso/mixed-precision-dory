

# Libraries
import numpy as np
import os

# DORY modules
import dory.Utils.Templates_writer.Network_template_writer as Network_writer
import dory.Utils.Templates_writer.Makefile_template_writer as Makefile_writer

from dory.Parsers.HW_node import HW_node


class Parser_HW_to_C:
    """
    Used to manage the generated source files. 
    
    Currently supporting: Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    """
    
    def __init__(
        self, 
        graph: list, 
        network_directory: str, 
        HW_description: dict[str, any], 
        verbose_level: str, 
        perf_layer: str, 
        save_string: str, 
        app_directory: str, 
        n_inputs: int = 1,
        num_cores: int = 8
    ):
        self.HWgraph = graph
        self.HW_description = HW_description
        self.verbose_level = verbose_level
        self.perf_layer = perf_layer
        self.save_string_for_Makefile = save_string
        self.network_directory = network_directory
        self.app_directory = app_directory
        self.inc_dir_rel = "inc"
        self.src_dir_rel = "src"
        self.hex_dir_rel = "hex"
        self.n_inputs = n_inputs
        self.num_cores = num_cores


    def adding_numbers_to_layers(self):
        for i, node in enumerate(self.HWgraph):
            node.name = node.name + str(i)


    def mapping_network_to_C_file(self):
        print("\nGenerating the .c file of the network.")
        Network_writer.print_template_network(
            self.HWgraph,
            self.HW_description,
            self.config_file,
            self.verbose_level,
            self.perf_layer,
            self.app_directory,
            self.inc_dir_rel,
            self.src_dir_rel
        )


    def mapping_makefile(self):
        print("\nGenerating the Makefile.")
        Makefile_writer.print_template_Makefile(
            self.HWgraph,
            self.HW_description,
            self.save_string_for_Makefile,
            self.app_directory
        )


    def l2_c_template(self, node, backend_library):
        if "Pool" in node.name:
            if(backend_library == '1D_Conv'):
                return "pooling_layer_1D_template.c"
            else:
                return "layer_L2_c_pooling_template.c"
        elif "Addition" in node.name:
            if(backend_library == '1D_Conv'):
                return "add_layer_1D_template.c"
            else:
                return "layer_L2_c_addition_template.c"
        else:
            return "layer_L2_c_conv_template.c.t"


    def l2_template_mapping(self, node, backend_library):
        tmpl_c = self.l2_c_template(node, backend_library)
        return {
            os.path.join(self.src_dir, node.prefixed_name + ".c"): os.path.join(self.tmpl_dir, tmpl_c),
            os.path.join(self.inc_dir, node.prefixed_name + ".h"): os.path.join(self.tmpl_dir, "layer_L2_h_template.h.t"),
        }


    def mapping_layers_to_C_files(self):
        print("\nTo be implemented in the target backend")


    def copy_backend_files(self, node):
        print("\nTo be implemented in the target backend")


    def copy_utils_files(self):
        print("\nCopying Utils.")
        for file in os.listdir(self.utils_files_dir):
            file_to_copy = os.path.join(self.utils_files_dir, file)
            if file_to_copy[-1] == 'c':
                os.system('cp -L "{}" {}'.format(file_to_copy, self.src_dir))
            elif file_to_copy[-1] == 'h':
                os.system('cp -L "{}" {}'.format(file_to_copy, self.inc_dir))


    def create_hex_weights_files(self):
        print("\nGenerating .hex weight files.")

        os.makedirs(self.hex_dir, exist_ok=True)

        for node_index, node in enumerate(self.HWgraph):
            constants = [None, None, None, None]

            for name in node.constant_names:
                lowered_name = name.lower()

                if "weight" in lowered_name:
                    constants[0] = name
                elif "bias" in lowered_name:
                    constants[1] = name
                elif name == "k":
                    constants[2] = name
                elif name == "l":
                    constants[3] = name

            flattened_values = []

            for key in constants:
                if key is None:
                    continue

                constant_data = node.__dict__.get(key)

                if constant_data is None or "value" not in constant_data:
                    raise KeyError(
                        f"Constant {key!r} is missing from node "
                        f"{node.prefixed_name!r}"
                    )

                value = np.asarray(constant_data["value"])

                if value.size == 0:
                    continue

                # Required because weights can be 4D while bias is 1D.
                flattened_values.append(value.reshape(-1))

            if not flattened_values:
                continue

            weights = np.concatenate(flattened_values)

            # Pad the total number of elements to a multiple of four.
            padding = (-weights.size) % 4

            if padding:
                weights = np.pad(
                    weights,
                    pad_width=(0, padding),
                    mode="constant",
                    constant_values=0,
                )

            output_name = f"{node.prefixed_name}_weights.hex"
            output_path = os.path.join(self.hex_dir, output_name)

            weights.astype(np.uint8).tofile(output_path)

            print(
                f"Saved {output_path}: "
                f"{weights.size} values, {weights.nbytes} source bytes"
            )


    def create_hex_input(self):
        print("\nGenerating .hex input file.")
        prefix = self.HWgraph[0].prefix
        for in_idx in range(self.n_inputs):
            infile = 'input.txt' if self.n_inputs == 1 else f'input_{in_idx}.txt'
            in_node = self.HWgraph[0]
            in_bits = in_node.input_activation_bits
            signed = in_node.input_activation_type == "int"
            try:
                x_in = np.loadtxt(
                    os.path.join(self.network_directory, infile), 
                    delimiter=',', 
                    dtype=np.uint8, 
                    usecols=[0]
                )
                x_in = x_in.flatten()
            except FileNotFoundError:
                print(f"========= WARNING ==========\n" \
                    f"Input file {os.path.join(self.network_directory, 'input.txt')} not found;\n" \
                    "generating random inputs!"
                )
                np.random.seed(42)
                x_in = np.random.randint(
                    low=-2**(in_node.input_activation_bits - 1) if signed else 0, 
                    high=2**(in_node.input_activation_bits - 1) - 1 if signed else 2**in_node.input_activation_bits,
                    size=self.HWgraph[0].group * self.HWgraph[0].input_channels * self.HWgraph[0].input_dimensions[0] * self.HWgraph[0].input_dimensions[1],
                    dtype=np.int8 if signed else np.uint8,
                )
                    
            if in_bits != 8:
                x_in = HW_node._compress(x_in, in_bits, signed)
            
            string_layer = prefix + "inputs.hex" if self.n_inputs == 1 else f"{prefix}inputs_{in_idx}.hex"
            save_s = os.path.join(self.hex_dir, string_layer)
            x_in.astype('uint8').tofile(save_s)
            

    @property
    def src_dir(self):
        return os.path.join(self.app_directory, self.src_dir_rel)


    @property
    def inc_dir(self):
        return os.path.join(self.app_directory, self.inc_dir_rel)


    @property
    def hex_dir(self):
        return os.path.join(self.app_directory, self.hex_dir_rel)


    def get_file_path(self):
        raise NotImplementedError("To be implemented by child class!")


    @property
    def tmpl_dir(self):
        return os.path.realpath(os.path.join(self.get_file_path(), 'Templates/layer_templates'))


    @property
    def utils_files_dir(self):
        return os.path.realpath(os.path.join(self.get_file_path(), 'Utils_files'))


    def full_graph_parsing(self):
        print("#####################################################")
        print("## DORY GENERAL PARSING FROM DORY HW IR TO C FILES ##")
        print("## FINAL RAPRESENTATION: COMPILABLE C PROJECT      ##")
        print("#####################################################")
        os.system('rm -rf {}'.format(self.app_directory))
        os.system('mkdir {}'.format(self.app_directory))
        os.system('mkdir {}'.format(self.src_dir))
        os.system('mkdir {}'.format(self.inc_dir))
        os.system('mkdir {}'.format(self.hex_dir))
        self.adding_numbers_to_layers()
        self.mapping_network_to_C_file()
        self.mapping_makefile()
        self.mapping_layers_to_C_files()
        self.copy_utils_files()
        self.create_hex_weights_files()
        self.create_hex_input()
        print("Done!")

