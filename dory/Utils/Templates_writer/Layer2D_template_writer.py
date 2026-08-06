import math
from mako.template import Template
import re
from collections import OrderedDict
import numpy as np
import sys
import os

from dory.Parsers.HW_node import HW_node



def _ceil_bits_to_bytes(bit_count: int) -> int:
    """Return storage bytes for an integer or symbolic bit count."""
    return (bit_count + 7) // 8


def _align_up(value: int, alignment: int = 8) -> int:
    """Round an integer byte offset up to a power-of-two alignment."""
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) // alignment * alignment


def _lut_memory_bytes(node: HW_node, safety_margin: int = 8) -> int:
    """Return the exact LUT allocation used by magnitude-indexed LUT kernels."""
    if node.implementation != "lut":
        return 0

    input_bits = int(node.input_activation_bits)
    weight_bits = int(node.weight_bits)
    if input_bits <= 0 or weight_bits <= 0:
        raise ValueError("LUT precisions must be positive")

    input_is_signed = node.input_activation_type == "int"
    input_entries = (1 << (input_bits - 1)) + 1 if input_is_signed else 1 << input_bits

    # LUT kernels index signed weights by magnitude: 0 .. 2^(bits-1).
    weight_entries = (1 << (weight_bits - 1)) + 1
    return input_entries * weight_entries * 4 + int(safety_margin)


def _standard_conv_im2col_bytes(
    num_cores: int, 
    kernel_h: str | int, 
    kernel_w: str | int, 
    tile_n_in: str | int
) -> int:
    """Scratch bytes required by standard PULP-NN convolution kernels."""
    return 2 * int(num_cores) * int(kernel_h) * int(kernel_w) * int(tile_n_in)


def _depthwise_im2col_bytes(
    num_cores: int, 
    kernel_h: str | int, 
    tile_n_in: str | int, 
    pad_top: str | int, 
    pad_bottom: str | int, 
    precision_parallelism: str | int
) -> int:
    """Scratch bytes required by the inherited depthwise kernel layout."""
    return (
        int(num_cores)
        * (int(kernel_h) * (int(tile_n_in) + int(pad_top) + int(pad_bottom)) + int(kernel_h))
        * int(precision_parallelism)
    )


def l2_layer_template(
    node: HW_node, 
    backend_lib: str, 
    double_buffering: int = 2, 
    num_cores: int = 8
) -> OrderedDict:
    """
    Generate a detailed template for a layer for code generation.
    Computes tiling sizes, memory offsets, strides, and other hardware-specific parameters.
    
    Args:
        node: Layer object with all required attributes.
        backend_lib: Library used by the source.
        double_buffering: Factor for double buffering, default 2.
        num_cores: number of available cores in the cluster
    Returns:
        tk: OrderedDict with all template parameters.
    """
    ks = node.kernel_shape
    s = node.strides
    g = node.group
    p = node.pads
    
    # compute convolution overlap for border handling
    conv_overlap_h = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
    conv_overlap1 = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
    conv_overlap2 = 2 * (ks[1] // 2) + ks[1] % 2 - 1 - (s[1] - 1)
    
    # padding
    padding_top, padding_left, padding_bottom, padding_right = p
    
    # layer naming
    name_layer = node.prefixed_name + '.h'
    
    tk = OrderedDict([])
    
    # identify the first layer
    if (re.search('.0', name_layer)):
        try:
            int(re.search('.0', name_layer).group())
            tk['first_layer'] = 0
        except ValueError:
            tk['first_layer'] = 1
    else:
        tk['first_layer'] = 0
        
    # misc falg and metadata
    tk['ULTRA_VERBOSE'] = False
    tk['verbose_log'] = ""
    tk['node'] = node
    tk['sdk'] = node.HW_description["software development kit"]["name"]
    tk['number_of_clusters'] = node.HW_description["HW specific parameters"].get("clusters", 1)
    tk['optional_type'] = backend_lib
    tk['func_name'] = node.prefixed_name
    tk['flag_DW'] = int(g > 1)
    tk['flag_LUT'] = int(node.implementation == "lut")
    tk['optional'] = node.op_type
    tk['FLAG_BATCHNORM'] = int('k' in node.constant_names)
    tk['has_bias'] = int(len([1 for name in node.constant_names if "bias" in name]) > 0)
    tk['FLAG_RELU'] = int('outshift' in node.constant_names)
    
    # data types
    tk['type'] = "float"
    if node.input_activation_type in ["int", "uint"]:
        tk['type'] = f"{node.input_activation_type}8_t" 
        
    # convolution overlap and padding
    tk['conv_overlap1'] = conv_overlap1
    tk['conv_overlap2'] = conv_overlap2
    tk['padding_top'] = padding_top
    tk['padding_bottom'] = padding_bottom
    tk['padding_left'] = padding_left
    tk['padding_right'] = padding_right
    tk['stride'] = s[0]

    # channel configuration
    if tk['flag_DW'] == 0:
        tk['g'] = 1
        tk['nif'] = node.tiling_dimensions["L2"]["input_dimensions"][0]
    else:
        tk['g'] = node.tiling_dimensions["L2"]["input_dimensions"][0]
        tk['nif'] = 1
        
    n_in = node.tiling_dimensions["L2"]["input_dimensions"][0]
    h_in = node.tiling_dimensions["L2"]["input_dimensions"][1]
    w_in = node.tiling_dimensions["L2"]["input_dimensions"][2]
    
    tile_n_in = node.tiling_dimensions["L1"]["input_dimensions"][0]
    tile_h_in = node.tiling_dimensions["L1"]["input_dimensions"][1]
    tile_w_in = node.tiling_dimensions["L1"]["input_dimensions"][2]

    if "Addition" not in node.name and "Pool" not in node.name:
        n_out  = node.tiling_dimensions["L2"]["weights_dimensions"][0]
    else:
        n_out = node.tiling_dimensions["L2"]["output_dimensions"][0]
        
    h_out = node.tiling_dimensions["L2"]["output_dimensions"][1]
    w_out = node.tiling_dimensions["L2"]["output_dimensions"][2]
    
    tile_n_out = node.tiling_dimensions["L1"]["output_dimensions"][0]
    tile_h_out = node.tiling_dimensions["L1"]["output_dimensions"][1]
    tile_w_out = node.tiling_dimensions["L1"]["output_dimensions"][2]
    
    # kernel and data shape
    fs1, fs2 = node.kernel_shape
    ds_x = node.input_activation_bits
    ds_y = node.output_activation_bits
    ds_act = node.constant_bits
    ds_W = node.weight_bits
    ds_bias = node.bias_bits

    dt_x = node.input_activation_type
    dt_y = node.output_activation_type
    dt_act = node.constant_type
    dt_W = node.weight_type

     # variant for addition layer
    if "Addition" in node.name:
        ds_x2 = node.second_input_activation_bits
        dt_x2 = node.second_input_activation_type
        tk["data_type_x2"] = dt_x2
        tk['x_data_size_byte2'] = ds_x2
        
        # layer specifi constants
        for suffix in ["inmul", "inadd", "inshift"]:
            for i in [1, 2]:
                tk[f"{suffix}{i}"] = node.__dict__[f"{suffix}{i}"]["value"]

        # output shift/mul/add
        tk["outmul"] = node.outmul["value"]
        tk["outadd"] = node.outadd["value"]
        tk["outshift"] = node.outshift["value"]

    if hasattr(node, "outmul"):
        tk['out_mul'] = node.outmul.get("value", 1)
    else:
        tk['out_mul'] = 1
    
    if hasattr(node, "outadd"):
        tk['out_add'] = node.outadd.get("value", 0)
    else:
        tk['out_add'] = 0
        
    if hasattr(node, "outshift"):
        tk['out_shift'] = node.outshift.get("value", 0)
    else:
        tk['out_shift'] = 0
        
    # tiling size
    DW = tk['flag_DW']
    has_bias = tk['has_bias']
    number_of_clusters = tk['number_of_clusters']

    tk["data_type_x"] = dt_x
    tk["data_type_y"] = dt_y
    tk["data_type_activations"] = dt_act
    tk["data_type_weights"] = dt_W
    tk['nof'] = n_out
    
    tk['factor'] = 1
    if node.HW_description['memory']['levels'] > 2:
        tk['factor'] = node.tiling_dimensions["L3"]["output_dimensions"][0] / n_out
        
    # input parameters
    tk['double_buffering'] = double_buffering
    tk['x_h'] = h_in
    tk['x_w'] = w_in
    tk['x_data_size_byte'] = ds_x
    tk['x_tile_size_nif'] = tile_n_in
    tk['x_tile_size_h'] = tile_h_in
    tk['x_tile_size_w'] = tile_w_in
    tk['x_tile_size_byte'] = int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
    tk['x_tile_size_nif_byte'] = int(math.ceil(tile_n_in * ds_x / 8.0))
    
    # output parameters
    tk['y_h'] = h_out
    tk['y_w'] = w_out
    tk['y_data_size_byte'] = ds_y
    tk['act_dim_bit'] = ds_act
    tk['y_tile_size_nof'] = tile_n_out if (n_out > tile_n_out) else n_out
    tk['y_tile_size_h'] = tile_h_out if (h_out > tile_h_out) > 0 else h_out
    tk['y_tile_size_w'] = tile_w_out if (w_out > tile_w_out) > 0 else w_out
    tk['y_tile_size_byte'] = int(math.ceil(tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w'] * ds_y / 8.0))
    tk['y_tile_size_nof_byte'] = int(math.ceil(tile_n_out * ds_y / 8.0))
    
    # strides and tile counts
    tk['x_stride_w_byte'] = int(math.ceil(w_in * n_in * ds_x / 8.0))
    tk['x_stride_c_byte'] = int(math.ceil(n_in * ds_x / 8.0))
    tk['y_stride_w_byte'] = int(math.ceil(w_out * n_out * tk['factor'] * ds_y / 8.0))
    tk['y_stride_c_byte'] = int(math.ceil(n_out * tk['factor'] * ds_y / 8.0))
    
    # tile count for each dimension
    tk['tile_dim_h'] = max(int(math.ceil(float(h_out) / float(tk['y_tile_size_h']))), 1)
    tk['tile_dim_w'] = max(int(math.ceil(float(w_out) / float(tk['y_tile_size_w']))), 1)
    tk['tile_dim_nof'] = max(int(math.ceil(float(n_out) / float(tk['y_tile_size_nof']))), 1)
    tk['tile_dim_nif'] = max(int(math.ceil(float(n_in) / float(tile_n_in))), 1)
    
    # last tile sizes
    tk['tile_n_in_last'] = n_in % tile_n_in if n_in % tile_n_in > 0 else tile_n_in
    tk['W_tile_size_nof_last'] = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
    
    # weight and bias parameters
    tk['fs1'], tk['fs2'] = fs1, fs2
    tk['W_data_size_byte'] = ds_W
    tk['b_data_size_byte'] = ds_bias
    tk['W_tile_size_nof'] = tile_n_out 
    
    # bias size (optional)
    tk['b_size_byte'] = 0
    if tk['has_bias'] == 1:
        tk['b_size_byte'] = int(math.ceil(n_out * ds_bias / 8.0))
        
    # weight tile size in input channels (depends on DW conv)
    tk['W_tile_size_nif'] = 1
    tk['W_tile_size_nif_last'] = 1
    if DW == 0:
        tk['W_tile_size_nif'] = tile_n_in * tk['tile_dim_nif']
        tk['W_tile_size_nif_last'] = tk['tile_n_in_last'] * tk['tile_dim_nif']    
        
    # compute weight related memory footprint
    if "Addition" not in node.name and "Pool" not in node.name:
        tk['W_tile_size_byte'] = int(
            math.ceil(tile_n_out * tk['W_tile_size_nif'] * fs1 * fs2 * ds_W / 8.0)
        )
        tk['W_stride_nof_byte'] = int(
            math.ceil(tk['nif'] * fs1 * fs2 * ds_W / 8.0)
        )        
        tk['W_stride_hw_byte'] = int(
            math.ceil(tk['nif'] * ds_W / 8.0)
        )
        tk['W_tile_nif_byte'] = int(
            math.ceil(tk['W_tile_size_nif'] * ds_W / 8.0)
        )
        tk['W_tile_nif_byte_last'] = int(
            math.ceil(tk['W_tile_size_nif_last'] * ds_W / 8.0)
        )
        
    # L2 memory offsets
    if tk['FLAG_BATCHNORM'] == 1:
        tk['l2_off_k'] = int(
            math.ceil(
                tk['nof'] * tk['nif'] * fs1 * fs2 * ds_W / 8.0 + tk['b_size_byte']
            )
        )
        tk['l2_off_lambda'] = int(
            math.ceil((tk['nof'] * tk['nif'] * fs1 * fs2 * ds_W + tk['nof'] * ds_act) / 8.0 + tk['b_size_byte'])
        )
    if has_bias == 1:
        tk['l2_off_bias'] = int(
            math.ceil(tk['nof'] * tk['nif'] * fs1 * fs2 * ds_W / 8.0)
        )
    
    # L1 buffer sizes and layout.  This is the single source of truth for
    # generated offsets; all sizes use ceiling byte conversion.
    safety_gap = 8
    valid_bits = [bits for bits in (ds_x, ds_y, ds_W) if bits is not None]
    precision_parallelism = max(1, 8 // min(valid_bits))

    single_x_buffer_size = int(_ceil_bits_to_bytes(ds_x * tile_n_in * tile_h_in * tile_w_in))
    x_is_tiled = not (n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in)
    x_buffer_size = single_x_buffer_size * (tk["double_buffering"] if x_is_tiled else 1)

    same_tiling_across_clusters = (
        (
            n_in == tile_n_in * number_of_clusters
            and w_in == tile_w_in
            and h_in == tile_h_in
            and n_out == tile_n_out * number_of_clusters
            and n_in > number_of_clusters
        )
        or (
            n_in == tile_n_in
            and w_in == tile_w_in
            and h_in == tile_h_in
            and n_out == tile_n_out * number_of_clusters
        )
    )
    buffer_factor = 1 if same_tiling_across_clusters else tk["double_buffering"]

    single_y_buffer_size = int(_ceil_bits_to_bytes(
        ds_y * tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w']
    ))
    y_buffer_size = buffer_factor * single_y_buffer_size

    if "Addition" not in node.name and "Pool" not in node.name:
        weight_nif = tk['W_tile_size_nif'] if DW == 0 else 1
        single_W_buffer_size = int(_ceil_bits_to_bytes(
            ds_W * tk['y_tile_size_nof'] * weight_nif * fs1 * fs2
        ))
        W_buffer_size = buffer_factor * single_W_buffer_size
    else:
        W_buffer_size = 0

    if tk['FLAG_BATCHNORM'] == 1:
        k_buffer_size = int(_ceil_bits_to_bytes(n_out * ds_act))
        lambda_buffer_size = k_buffer_size
    else:
        k_buffer_size = 0
        lambda_buffer_size = 0

    tk['k_tile_size_byte'] = 0
    tk['lambda_tile_size_byte'] = 0
    tk['k_size_byte'] = 0
    tk['lambda_size_byte'] = 0
    tk['k_tile_size_byte_transfer'] = 0
    tk['lambda_tile_size_byte_transfer'] = 0

    if "Pool" not in node.name and tk['FLAG_BATCHNORM'] == 1:
        tk['k_size_byte'] = k_buffer_size
        tk['lambda_size_byte'] = lambda_buffer_size
        tk['k_tile_size_byte_transfer'] = int(_ceil_bits_to_bytes(tile_n_out * ds_act))
        tk['lambda_tile_size_byte_transfer'] = int(_ceil_bits_to_bytes(tile_n_out * ds_act))

        bn_factor = (
            1
            if n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in and n_out == tile_n_out
            else tk['double_buffering']
        )
        tk['k_tile_size_byte'] = bn_factor * int(_ceil_bits_to_bytes(tile_n_out * ds_act))
        tk['lambda_tile_size_byte'] = bn_factor * int(_ceil_bits_to_bytes(tile_n_out * ds_act))

    if "Pool" not in node.name and has_bias == 1:
        tk['bias_tile_size_byte'] = int(_ceil_bits_to_bytes(tile_n_out * ds_bias))
        tk['b_size_byte'] = int(_ceil_bits_to_bytes(n_out * ds_bias))
    else:
        tk['bias_tile_size_byte'] = 0
        tk['b_size_byte'] = 0
        
    # Sequential base-buffer layout.  Preserve the template's historical 8-byte
    # guard between x, y, W, k, lambda, and bias slots, including empty slots.
    if "Addition" not in node.name and "Pool" not in node.name:
        tk['l1_x_offset'] = 0
        tk['l1_y_offset'] = tk['l1_x_offset'] + x_buffer_size + safety_gap
        tk['l1_W_offset'] = tk['l1_y_offset'] + y_buffer_size + safety_gap
        tk['l1_k_offset'] = tk['l1_W_offset'] + W_buffer_size + safety_gap
        tk['l1_lambda_offset'] = tk['l1_k_offset'] + tk['k_tile_size_byte'] + safety_gap
        tk['l1_b_offset'] = tk['l1_lambda_offset'] + tk['lambda_tile_size_byte'] + safety_gap
        buffer_l1_all = tk['l1_b_offset'] + tk['b_size_byte']

        if DW == 0:
            tk['im2col_dim'] = _standard_conv_im2col_bytes(
                num_cores, fs1, fs2, tile_n_in
            )
        else:
            tk['im2col_dim'] = _depthwise_im2col_bytes(
                num_cores, fs1, tile_n_in, padding_top, padding_bottom, precision_parallelism
            )

        if "FullyConnected" in node.name:
            tk['im2col_dim'] = 0

        tk['lut_dim'] = _lut_memory_bytes(node, safety_margin=safety_gap)
        tk['l1_lut_offset'] = buffer_l1_all
        tk['l1_im2col_offset'] = buffer_l1_all + tk['lut_dim']
        tk['buffer_l1_total'] = tk['l1_im2col_offset'] + tk['im2col_dim']

    elif "Addition" in node.name:
        tk['l1_x_offset'] = 0
        tk['l1_y_offset'] = x_buffer_size + safety_gap
        tk['l1_x2_offset'] = tk['l1_y_offset'] + y_buffer_size + safety_gap
        buffer_l1_all = sum((
            x_buffer_size * tk['double_buffering'],
            y_buffer_size,
            tk['k_tile_size_byte'],
            tk['lambda_tile_size_byte'],
            40,
            tk['b_size_byte'],
        ))
        tk['im2col_dim'] = 0
        tk['lut_dim'] = 0
        tk['buffer_l1_total'] = buffer_l1_all

    else:  # Pool
        tk['l1_x_offset'] = 0
        tk['l1_y_offset'] = x_buffer_size + safety_gap
        buffer_l1_all = sum((
            x_buffer_size,
            y_buffer_size,
            tk['k_tile_size_byte'],
            tk['lambda_tile_size_byte'],
            40,
            tk['b_size_byte'],
        ))
        tk['im2col_dim'] = 0
        tk['lut_dim'] = 0
        tk['buffer_l1_total'] = buffer_l1_all

    tk['buffer_l1_all'] = buffer_l1_all

    # handle last tile case
    if "Addition" not in node.name and "Pool" not in node.name:
        tk['W_tile_size_nof_last'] = n_out % tile_n_out if n_out % tile_n_out > 0 else tile_n_out
        tk['W_tile_size_nif_last'] = tk['W_tile_size_nif']
        tk['W_tile_size_nif_byte_last'] = int(
            _ceil_bits_to_bytes(tk['W_tile_size_nif_last'] * ds_W)
        )

    tk['y_tile_size_nof_last'] = n_out % tile_n_out if n_out % tile_n_out > 0 else tile_n_out
    tk['y_tile_size_h_last'] = h_out % tile_h_out if h_out % tile_h_out > 0 else tile_h_out
    tk['y_tile_size_w_last'] = w_out % tile_w_out if w_out % tile_w_out > 0 else tile_w_out
    tk['y_length_nof_byte_last'] = int(
        _ceil_bits_to_bytes(tk['y_tile_size_nof_last'] * ds_y)
    )

    tk['x_tile_size_nif_last'] = n_in % tile_n_in if n_in % tile_n_in > 0 else tile_n_in
    tk['x_tile_size_nif_byte_last'] = int(_ceil_bits_to_bytes(tk['x_tile_size_nif_last'] * ds_x))
    tk['x_tile_size_h_last'] = tk['y_tile_size_h_last'] * s[0] + ks[0] - s[0] - (
        padding_bottom - ((h_in + padding_bottom + padding_top) - (h_out * s[0] + ks[0] - s[0]))
    )
    tk['x_tile_size_w_last'] = tk['y_tile_size_w_last'] * s[1] + ks[1] - s[1] - (
        padding_right - ((w_in + padding_left + padding_right) - (w_out * s[1] + ks[1] - s[1]))
    )
    tk['x_tile_size_h_last'] = min(tk['x_tile_size_h_last'], tk['x_tile_size_h'])
    tk['x_tile_size_w_last'] = min(tk['x_tile_size_w_last'], tk['x_tile_size_w'])

    # flags
    tk['conv1d'] = node.conv1d
    tk['dilations'] = node.dilations

    # logging debug info
    log_str = ""
    for k, v in tk.items():
        log_str += f"// {k.ljust(30)} {v}\n"
    tk['verbose_log'] = log_str

    return tk



def l3_layer_template(node: HW_node, num_cores: int = 8) -> OrderedDict:
    """
    Populate the template of the layer to move the data between L3 and L2
    

    Args:
        node (HW_node): Layer reppresentation,
        num_cores (int): number of available cores in the cluster.

    Returns:
        tk: dictionary with the arguments necessary for the template
    """
    ks = node.kernel_shape
    s = node.strides
    g = node.group
    p = node.pads
    padding_top, padding_left, padding_bottom, padding_right = p[0], p[1], p[2], p[3] 
    conv_overlap1 = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
    conv_overlap2 = 2 * (ks[1] // 2) + ks[1] % 2 - 1 - (s[1] - 1)
    tk = OrderedDict([])
    
    # FLAGS
    # check depth-wise convolutions
    tk['flag_DW'] = int(g > 1)
    tk['ULTRA_VERBOSE'] = False
    
    ds_x = node.input_activation_bits
    h_out = node.tiling_dimensions["L3"]["output_dimensions"][1]
    ds_y = node.output_activation_bits


    n_in_L2 = node.tiling_dimensions["L2"]["input_dimensions"][0]
    if node.tiling_dimensions["L3"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
        h_in_L2 = node.tiling_dimensions["L2"]["output_dimensions"][1] * s[0] + (ks[0] - 1) - (s[0] - 1)
    else:
        h_in_L2 = node.tiling_dimensions["L2"]["input_dimensions"][1]
        
    w_in_L2 = node.tiling_dimensions["L2"]["input_dimensions"][2]
    
    if "Addition" not in node.name and "Pool" not in node.name:
        n_out_L2 = node.tiling_dimensions["L2"]["weights_dimensions"][0]
    else:
        n_out_L2 = node.tiling_dimensions["L2"]["output_dimensions"][0]
    if node.tiling_dimensions["L3"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
        h_out_L2 = int(
            np.floor(
                (node.tiling_dimensions["L2"]["input_dimensions"][1] - (ks[0] - 1) + (s[0] - 1)) / s[0]
            )
        )
    else:
        h_out_L2 = node.tiling_dimensions["L2"]["output_dimensions"][1]
            
    w_out_L2 = node.tiling_dimensions["L2"]["output_dimensions"][2]
    
    tk['conv_overlap1'] = conv_overlap1
    tk['conv_overlap2'] = conv_overlap2
    tk['padding'] = padding_top
    if (node.L3_input):
        tk['input_L3'] = 1
        factor_h_in = int(h_out / h_out_L2) 
    else:
        tk['input_L3'] = 0
        factor_h_in = 1
        
    factor_h_out = int(
        node.tiling_dimensions["L3"]["output_dimensions"][1] / node.tiling_dimensions["L2"]["output_dimensions"][1]
    )
    if not isinstance(node.tiling_dimensions["L2"]["weights_dimensions"], type(None)):
        factor_ch_out = int(
            node.tiling_dimensions["L3"]["weights_dimensions"][0] / node.tiling_dimensions["L2"]["weights_dimensions"][0]
        )
    else:
        factor_ch_out = 1
    
    tk['n_tile_W'] = factor_ch_out
    tk['n_tile_x'] = factor_h_in
    tk['n_tile_y'] = factor_h_out
    
    tk['verbose'] = False
    if tk['padding'] > 0:
        tk['func_name'] = [node.prefixed_name + "_L2", node.prefixed_name + "_L2_p_t", node.prefixed_name + "_L2_p_b"]
    else:
        tk['func_name'] = [node.prefixed_name + "_L2"]
        
    tk['func_name_L3'] = node.prefixed_name
    tk['BitIn'] = ds_x
    tk['y_data_size_byte'] = ds_y
    tk['x_data_size_byte'] = ds_x
    tk['w_out'] = w_out_L2
    tk['h_out'] = h_out_L2
    tk['n_out'] = n_out_L2
    tk['w_in'] = w_in_L2
    tk['h_in'] = h_in_L2
    tk['n_in'] = n_in_L2

    tk['has_bias'] = int(
        len([1 for name in node.constant_names if "bias" in name]) > 0
    )

    offset = 0
    tk['l3_offset_w'] = offset
    offset += node.tiling_dimensions["L3"]["weight_memory"]

    if tk['has_bias'] == 1:
        tk['l3_offset_b'] = offset
        offset += node.tiling_dimensions["L3"]["bias_memory"]

    if not isinstance(node.tiling_dimensions["L2"]["constants_memory"], type(None)):
        tk['l3_offset_k'] = offset
        offset += int(node.tiling_dimensions["L3"]["constants_memory"] / 2)

        tk['l3_offset_l'] = offset
        offset += int(node.tiling_dimensions["L3"]["constants_memory"] / 2)

    tk['weight_dim'] = int( node.tiling_dimensions["L2"]["weight_memory"] )
    if tk['has_bias'] == 1:
        tk['bias_dim'] = node.tiling_dimensions["L2"]["bias_memory"]
    else:
        tk['bias_dim'] = 0
    if not isinstance(node.tiling_dimensions["L2"]["constants_memory"], type(None)):
        tk['lambda_dim'] = int(node.tiling_dimensions["L2"]["constants_memory"] / 2)
        tk['k_dim'] = int(node.tiling_dimensions["L2"]["constants_memory"] / 2)
    else:
        tk['lambda_dim'] = 0
        tk['k_dim'] = 0
        
    tk['dim_out'] = int( n_out_L2 * w_out_L2 * h_out_L2 * node.output_activation_bits / 8 )
    tk['dim_in'] = int( n_in_L2 * w_in_L2 * h_in_L2 * node.input_activation_bits / 8 )

    tk['verbose_log'] = ""

    return tk
    