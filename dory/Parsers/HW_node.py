import numpy as np
import copy
import os
from .DORY_node import DORY_node
from .Layer_node import Layer_node


class HW_node(DORY_node):

    Tiler = None

    def __init__(self, node: Layer_node, HW_description: dict) -> None:
        super().__init__()
        self.__dict__ = node.__dict__
        self.tiling_dimensions = {}
        lvl = None
        for level in range(HW_description["memory"]["levels"]):
            lvl = f"L{level + 1}"
            self.tiling_dimensions[lvl] = {}
            self.tiling_dimensions[lvl]["weights_dimensions"] = None
            self.tiling_dimensions[lvl]["input_dimensions"] = None
            self.tiling_dimensions[lvl]["output_dimensions"] = None
            self.tiling_dimensions[lvl]["weight_memory"] = None
            self.tiling_dimensions[lvl]["bias_memory"] = None
            self.tiling_dimensions[lvl]["constants_memory"] = None
            self.tiling_dimensions[lvl]["input_activation_memory"] = None
            self.tiling_dimensions[lvl]["output_activation_memory"] = None
            
        if not isinstance(self.name, type(None)):
            if "Convolution" in self.name or "FullyConnected" in self.name:
                self.tiling_dimensions[lvl]["weights_dimensions"] = [self.output_channels, self.input_channels]
        
        self.tiling_dimensions[lvl]["input_dimensions"] = [self.input_channels] + self.input_dimensions
        self.tiling_dimensions[lvl]["output_dimensions"] = [self.output_channels] + self.output_dimensions
        self.tiling_dimensions[lvl]["weight_memory"] = self.weight_memory
        self.tiling_dimensions[lvl]["bias_memory"] = self.bias_memory
        self.tiling_dimensions[lvl]["constants_memory"] = self.constants_memory
        self.tiling_dimensions[lvl]["input_activation_memory"] = self.input_activation_memory
        self.tiling_dimensions[lvl]["output_activation_memory"] = self.output_activation_memory
        
        self.HW_description = HW_description
        self.check_sum_w = None
        self.check_sum_in = None
        self.check_sum_out = None
        self.L3_input = 0
        
        try:
            self.split_ints = HW_description['split_ints']
        except KeyError:
            self.split_ints = False


    def create_tiling_dimensions(
        self,
        previous_node: DORY_node,
        config_file: dict,
        num_cores: int = 8,
        input_nodes: list[DORY_node] = None,
    ) -> None:
        
        for level in range(self.HW_description["memory"]["levels"], 1, -1):
            
            weights_dim, input_dims, output_dims = self.Tiler(
                self,
                previous_node,
                config_file["code reserved space"],
                num_cores=num_cores,
                input_HW_nodes=input_nodes,
            ).get_tiling(level)
            
            self.tiling_dimensions[f"L{level-1}"]["input_dimensions"] = input_dims
            self.tiling_dimensions[f"L{level-1}"]["output_dimensions"] = output_dims
            
            if "Convolution" in self.name or "FullyConnected" in self.name:
                self.tiling_dimensions[f"L{level-1}"]["weights_dimensions"] = weights_dim

                groups = (
                    self.group 
                    if all(self.group <= d for d in weights_dim) 
                    else max(weights_dim)
                )

                self.tiling_dimensions[f"L{level-1}"]["weight_memory"] = (
                    np.prod(weights_dim) 
                    / groups 
                    * np.prod(self.kernel_shape) 
                    * self.weight_bits 
                    / 8
                )
                
                lut_dim = 0
                if self.implementation == "lut" and (level - 1) == 1:
                    lut_dim = (
                        2 ** (
                            self.weight_bits 
                            + self.input_activation_bits
                            ) 
                        * self.bias_bits
                    )
                self.tiling_dimensions[f"L{level-1}"]["lut_memory"] = lut_dim
                
            else:
                self.tiling_dimensions[f"L{level-1}"]["weight_memory"] = 0
                
            constants_memory = 0
            bias_memory = 0
            
            for name in self.constant_names:
                if name in ["l","k"]:
                    constants_memory += weights_dim[0] * self.constant_bits / 8
                if "bias" in name:
                    if groups == 1:
                        bias_memory += weights_dim[0] * self.bias_bits / 8
                    else:
                        bias_memory += weights_dim[0] * self.bias_bits / 8 * 16

            
            self.tiling_dimensions[f"L{level-1}"]["bias_memory"] = int(bias_memory)
            self.tiling_dimensions[f"L{level-1}"]["constants_memory"] = int(constants_memory)
            self.tiling_dimensions[f"L{level-1}"]["input_activation_memory"] = np.prod(
                self.tiling_dimensions["L{}".format(level-1)]["input_dimensions"]
            ) * self.input_activation_bits / 8
            
            self.tiling_dimensions[f"L{level-1}"]["output_activation_memory"] = np.prod(
                self.tiling_dimensions["L{}".format(level-1)]["output_dimensions"]
            ) * self.output_activation_bits / 8


    def rename_weights(self) -> None:
        if "Convolution" in self.name or "FullyConnected" in self.name:
            for i, name in enumerate(self.constant_names):
                if name not in ["l", "k", "outshift", "outmul", "outadd"]:
                    if "bias" not in name:
                        if len(self.__dict__[name]["value"].flatten()) > self.output_channels:
                            self.__dict__["weights"] = self.__dict__.pop(name)
                            self.constant_names[i] = "weights"


    @staticmethod
    def _compress(x: np.ndarray, bits: int, signed: bool = False) -> np.ndarray:
        """
        Packs an array of integers (x) into bytes, supporting signed or unsigned formats.

        Args:
            x (np.ndarray): Input array of integer values (e.g. int8, int16, etc.)
            bits (int): Number of bits per element (2, 4, 8)
            signed (bool): Whether the input values are signed (two’s complement)

        Returns:
            np.ndarray (dtype=np.uint8): Packed byte array
        """
        n_elements_in_byte = 8 // bits
        max_val = 2**(bits - 1) - 1 if signed else 2**bits - 1
        min_val = -2**(bits - 1) if signed else 0

        x = np.clip(x, min_val, max_val).astype(np.int32)

        if signed:
            x = np.where(x < 0, x + (1 << bits), x)

        x_masked = x & ((1 << bits) - 1)
        x_reshaped = x_masked.reshape((-1, n_elements_in_byte))

        po2 = 2 ** (np.arange(n_elements_in_byte) * bits)
        po2 = np.tile(po2, (x_reshaped.shape[0], 1))

        packed = np.sum(x_reshaped * po2, axis=1).astype(np.uint8)

        return packed

    @staticmethod
    def _to_uint8(x: np.ndarray, bits: int) -> np.ndarray:
        n_mult = bits//8
        x = np.tile(x[:, None], (1, n_mult))
        shifts = np.tile(8 * np.arange(n_mult), (x.shape[0], 1))
        x_shift_masked = (x >> shifts) & 255
        x_flat = x_shift_masked.ravel().astype(np.uint8)
        return x_flat


    def add_checksum_w_integer(self) -> None:
        self.check_sum_w = 0
        weight_name = ""
        
        if "Convolution" in self.name or "FullyConnected" in self.name:
            for name in self.constant_names:
                if name not in ["l", "k", "outshift","outmul","outadd"]:
                    if "bias" not in name:
                        weight_name = name
                        
        if weight_name in self.__dict__:
            if self.weight_bits < 8 and self.group > 1:
                self.__dict__[weight_name]["value"] = np.asarray(self.__dict__[weight_name]["value"])
                self.__dict__[weight_name]["value"] = self.__dict__[weight_name]["value"].reshape(
                    self.__dict__[weight_name]["value"].shape[0] // 2, 
                    2, 
                    self.__dict__[weight_name]["value"].shape[1],
                    self.__dict__[weight_name]["value"].shape[2],
                    self.__dict__[weight_name]["value"].shape[3]
                ).transpose(0,2,3,1,4).flatten()
            else:
                self.__dict__[weight_name]["value"] = self.__dict__[weight_name]["value"].flatten()
            
            signed = self.__dict__[weight_name]["value"].min() < 0.0
            self.__dict__[weight_name]["value"] = self.__dict__[weight_name]["value"].astype(np.int8 if signed else np.uint8)
            
            if self.weight_bits != 8:
                self.__dict__[weight_name]["value"] = self._compress(
                    self.__dict__[weight_name]["value"], 
                    self.weight_bits, 
                    signed
                )
                
            self.check_sum_w += sum(self.__dict__[weight_name]["value"])

        bias_name = ""
        if "Convolution" in self.name or "FullyConnected" in self.name:
            for name in self.constant_names:
                if name not in ["l", "k", "outshift", "outmul", "outadd"]:
                    if "bias" in name:
                        bias_name = name

        if bias_name in self.__dict__:
            self.__dict__[bias_name]["value"] = self._to_uint8(
                self.__dict__[bias_name]['value'].astype(np.int64).ravel(), 
                self.bias_bits
            )
            self.check_sum_w += sum(self.__dict__[bias_name]["value"])

        if 'k' in self.__dict__:
            self.k["value"] = self._to_uint8(
                self.k['value'].astype(np.int64).ravel(), 
                self.constant_bits
            )
            self.check_sum_w += sum(self.k["value"])

        if 'l' in self.__dict__:
            self.l["value"] = self._to_uint8(
                self.l['value'].astype(np.int64).ravel(), 
                self.constant_bits
            )
            self.check_sum_w += sum(self.l["value"])


    def add_checksum_activations_integer(self, load_directory: str, node_number: int, n_inputs: int = 1) -> None:
        self.check_sum_in = []
        self.check_sum_out = []
        for in_idx in range(n_inputs):
            if node_number == 0:
                infile = 'input.txt' if n_inputs == 1 else f'input_{in_idx}.txt'
                try:
                    try:
                        x = np.loadtxt(
                            os.path.join(load_directory, infile), 
                            delimiter=',', 
                            dtype=np.uint8, 
                            usecols=[0]
                        )
                    except ValueError:
                        x = np.loadtxt(
                            os.path.join(load_directory, infile), 
                            delimiter=',', 
                            dtype=np.float, 
                            usecols=[0]
                        ).astype(np.int64)
                    
                    x = x.ravel()
                    if self.input_activation_bits <= 8:
                        x = self._compress(x, self.input_activation_bits)
                        
                except FileNotFoundError:
                    print("========= WARNING ==========")
                    print(f"Input file {os.path.join(load_directory, 'input.txt')} not found; generating random inputs!")
                    x = np.random.randint(
                        low=0, 
                        high=2**8 - 1,
                        size=self.input_channels * self.input_dimensions[0] * self.input_dimensions[1],
                        dtype=np.uint8
                    )
            else:
                infile = f'out_layer{node_number-1}.txt' if n_inputs == 1 else f'out_{in_idx}_layer{node_number-1}.txt'
                try:
                    x = np.loadtxt(
                        os.path.join(load_directory, infile), 
                        delimiter=',', 
                        dtype=np.int64, 
                        usecols=[0]
                    )
                except ValueError:
                    x = np.loadtxt(
                        os.path.join(load_directory, infile), 
                        delimiter=',', 
                        dtype=np.float, 
                        usecols=[0]
                    ).astype(np.int64)
                    
                if self.input_activation_bits <= 8:
                    x = self._compress(x.ravel(), self.input_activation_bits)

            self.check_sum_in.append(int(sum(x)))
            outfile = f'out_layer{node_number}.txt' if n_inputs == 1 else f'out_{in_idx}_layer{node_number}.txt'
            try:
                y = np.loadtxt(
                    os.path.join(load_directory, outfile), 
                    delimiter=',', 
                    dtype=np.int64, 
                    usecols=[0]
                )
            except ValueError:
                y = np.loadtxt(
                    os.path.join(load_directory, outfile), 
                    delimiter=',', 
                    dtype=np.float, 
                    usecols=[0]
                ).astype(np.int64)
                
            if self.output_activation_bits <= 8:
                y = self._compress(y.ravel(), self.output_activation_bits)
            elif self.split_ints and self.output_activation_bits > 8:
                y = self._to_uint8(y.ravel(), self.output_activation_bits)

            self.check_sum_out.append(int(y.sum()))

    def export_to_dict(self) -> None:
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
            
            elif (
                not isinstance(value, dict) 
                and key != "name" 
                and key in Layer_node().__dict__.keys()
            ):
                node_dict["Layer_node_parameters"][key] = value
            
            elif key == "tiling_dimensions":
                node_dict["Tiling_parameters"] = {}
                for key1, value1 in value.items():
                    node_dict["Tiling_parameters"][key1] = {}
                    for key2, value2 in value1.items():
                        node_dict["Tiling_parameters"][key1][key2] = value2
            
            elif key in self.constant_names:
                node_dict["Weights"][key] = {}
                node_dict["Weights"][key]["Present"] = 'Yes'
                node_dict["Weights"][key]["Layout"] = value["layout"]
                
        return node_dict
