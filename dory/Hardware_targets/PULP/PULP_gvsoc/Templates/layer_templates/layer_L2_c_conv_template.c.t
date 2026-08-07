

#include "${func_name}.h"
% if sdk == 'gap_sdk':
#include "pulp.h"
% endif
#include "pmsis.h"
#include "dory_get_tile.h"
#include "dory_dma.h"
#include "pulp_nn_kernels.h"

% if ULTRA_VERBOSE:
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
% endif

void ${func_name}(void *args) 
{
  // pointers and offsets of buffers in L2 and L3
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int l3_x = (unsigned int) real_arg[0];
  unsigned int l3_y = (unsigned int) real_arg[1];
  unsigned int l3_W = (unsigned int) real_arg[2];
  unsigned int l2_x = (unsigned int) real_arg[3];
  unsigned int l2_x_2 = (unsigned int) real_arg[4];
  unsigned int l2_y = (unsigned int) real_arg[5];
  unsigned int l2_W = (unsigned int) real_arg[6];
  unsigned int l1_buffer = (${type} *) real_arg[7];
  unsigned int hyperram = (unsigned int) real_arg[8];
  unsigned int out_mult_in = (unsigned int) real_arg[9];
  unsigned int out_shift_in = (unsigned int) real_arg[10];
% if ULTRA_VERBOSE:
  if (pi_core_id() == 0)
  {
    print_layer_args(real_arg);
  }
  
% endif

  // define DMA structure for data transfer
  volatile DMA_copy DMA_copy_k, DMA_copy_lambda;
  volatile DMA_copy DMA_copy_W, DMA_copy_x, DMA_copy_y;
  // init DMA structures
  int dma_id = dory_dma_allocate();

% if has_bias == 1:
  volatile DMA_copy DMA_copy_bias;
  DMA_copy_bias.hwc_to_chw = 0;
  DMA_copy_bias.stride_2d = 0;
  DMA_copy_bias.stride_1d = 0;
  DMA_copy_bias.dir = 1;
  DMA_copy_bias.tid = dma_id;
  
% endif
  DMA_copy_k.hwc_to_chw = 0;
  DMA_copy_k.stride_2d = 0;
  DMA_copy_k.stride_1d = 0;
  DMA_copy_k.dir = 1;
  DMA_copy_k.tid = dma_id;

  DMA_copy_lambda.hwc_to_chw = 0;
  DMA_copy_lambda.stride_2d = 0;
  DMA_copy_lambda.stride_1d = 0;
  DMA_copy_lambda.dir = 1;
  DMA_copy_lambda.tid = dma_id;

% if flag_DW == 1:
  DMA_copy_x.hwc_to_chw = 1;
% else:
  DMA_copy_x.hwc_to_chw = 0;
% endif
  DMA_copy_x.stride_2d = ${x_stride_w_byte};
  DMA_copy_x.stride_1d = ${x_stride_c_byte};
  DMA_copy_x.dir = 1;
  DMA_copy_x.tid = dma_id;

  DMA_copy_W.hwc_to_chw = 0;
  DMA_copy_W.stride_2d = ${W_stride_nof_byte};
  DMA_copy_W.stride_1d = ${W_stride_hw_byte};
  DMA_copy_W.dir = 1;
  DMA_copy_W.tid = dma_id;

  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = ${y_stride_w_byte};
  DMA_copy_y.stride_1d = ${y_stride_c_byte};
  DMA_copy_y.dir = 0;     // L1 -> L2
  DMA_copy_y.tid = dma_id;

  // how much data fits per iteration and pointers to currnet buffers in L1
  volatile int p_r, p_l, p_t, p_b;
% if tile_dim_nif * tile_dim_h * tile_dim_w != 1:
  volatile unsigned short x_tile_size_nif;
  volatile unsigned short x_tile_size_h;
  volatile unsigned short x_tile_size_w;
  volatile unsigned short x_tile_size_byte;
  volatile unsigned short x_length_nif_byte;
  volatile int pad_offset_h, pad_offset_w;
% endif
  volatile unsigned short W_tile_size_nof;
  volatile unsigned short W_tile_size_nif;
  volatile unsigned short W_tile_size_byte;
  volatile unsigned short W_length_nif_byte;
  volatile ${type} *x, *W, *y, *b;
% if FLAG_BATCHNORM == 1:
% if act_dim_bit == 32:
  volatile int32_t *k;
  volatile int32_t *lambda;
% else:
  volatile int64_t *k;
  volatile int64_t *lambda;
% endif
% endif
  volatile int x_tile_size_nif_exec;
  volatile int x_tile_size_h_exec;
  volatile int x_tile_size_w_exec;
  volatile int y_tile_size_nof;
  volatile int y_tile_size_h;
  volatile int y_tile_size_w;
  volatile int y_tile_size_byte;
  volatile int y_length_nof_byte;
  volatile int db_x;
  volatile int db_W;
  volatile int db_act;
  volatile int db_y;
  volatile int exec_db_x;
  volatile int exec_db_W;
  volatile int exec_db_act;
  // double buffering state
  int db_state_x = 0;
  int db_state_W = 0;
  int db_state_y = 1;
  // last-tile flags
  int iter;
  // tile loop indeces
  int _i_nof_load = 0, _i_nif_load = 0, _i_h_load = 0, _i_w_load = 0;
  int _i_nof_exec = 0, _i_nif_exec = 0, _i_h_exec = 0, _i_w_exec = 0;
% if has_bias == 1:
  int has_bias = 1;
% endif
  // im2col buffer pointer
  volatile ${type} *im2col;
  im2col = l1_buffer + ${buffer_l1_all + lut_dim};
% if flag_LUT:
  // LUT buffer pointer
  volatile ${type} *lut;
  lut = l1_buffer + ${buffer_l1_all};
% endif
% if flag_DW == 1:
  // depth-wise buffer pointer
  volatile ${type} *pwt_buffer;
  pwt_buffer = im2col + ${im2col_dim};
% endif
% if FLAG_RELU == 1:
  // quantization scale
  uint16_t out_mult = out_mult_in;
% endif
  // rescale
  uint16_t out_shift = out_shift_in;

% if has_bias == 1:
  // copy bias from L2 to L1
  DMA_copy_bias.ext = (uint32_t) l2_W + ${l2_off_bias};
  DMA_copy_bias.loc = (uint32_t) (l1_buffer + ${l1_b_offset});
  DMA_copy_bias.number_of_2d_copies = 1;
  DMA_copy_bias.number_of_1d_copies = 1;
  DMA_copy_bias.length_1d_copy = (uint16_t) ${b_size_byte};
  dory_dma_memcpy_async(&DMA_copy_bias);
  dory_dma_barrier(&DMA_copy_bias);
% if ULTRA_VERBOSE:
  if (pi_core_id() == 0)
  {
    print_DMA_transfer(&DMA_copy_bias, "Bias", l2_W, l1_buffer);
  }
% endif

% endif
% if FLAG_BATCHNORM == 1:
  // copy BN parameters frm L2 to L1
  DMA_copy_k.ext = (uint32_t) l2_W + ${l2_off_k};
  DMA_copy_k.loc = (uint32_t) l1_buffer + ${l1_k_offset};
  DMA_copy_k.number_of_2d_copies = 1;
  DMA_copy_k.number_of_1d_copies = 1;
  DMA_copy_k.length_1d_copy = (uint16_t) ${k_tile_size_byte_transfer};
  dory_dma_memcpy_async(&DMA_copy_k);
  dory_dma_barrier(&DMA_copy_k);

  DMA_copy_lambda.ext = (uint32_t) l2_W + ${l2_off_lambda};
  DMA_copy_lambda.loc = (uint32_t) l1_buffer + ${l1_lambda_offset};
  DMA_copy_lambda.number_of_2d_copies = 1;
  DMA_copy_lambda.number_of_1d_copies = 1;
  DMA_copy_lambda.length_1d_copy = (uint16_t) ${lambda_tile_size_byte_transfer};
  dory_dma_memcpy_async(&DMA_copy_lambda);
  dory_dma_barrier(&DMA_copy_lambda);

% endif
  // copy first input from L2 to L1
  DMA_copy_x.ext = l2_x;
  DMA_copy_x.loc = (l1_buffer + ${l1_x_offset}) + 0;
  DMA_copy_x.number_of_2d_copies = ${x_tile_size_h};
  DMA_copy_x.number_of_1d_copies = ${x_tile_size_w};
  DMA_copy_x.length_1d_copy = ${x_tile_size_nif_byte};
  dory_dma_memcpy_async(&DMA_copy_x);
  dory_dma_barrier(&DMA_copy_x);
% if ULTRA_VERBOSE:
  if (pi_core_id() == 0)
  {
    print_DMA_transfer(&DMA_copy_x, "Input", l2_x, l1_buffer);
    debug_print_tensor(&DMA_copy_x, "Input", 256);
  }
% endif

  // copy weights from L2 to L1
  DMA_copy_W.ext = l2_W;
  DMA_copy_W.loc = (l1_buffer + ${l1_W_offset}) + 0;
% if flag_DW == 0:
  DMA_copy_W.number_of_2d_copies = ${W_tile_size_nof};
  DMA_copy_W.number_of_1d_copies = ${fs1 * fs2};
  DMA_copy_W.length_1d_copy = ${W_tile_nif_byte};
% else:
  DMA_copy_W.number_of_2d_copies = 1;
  DMA_copy_W.number_of_1d_copies = 1;
  DMA_copy_W.length_1d_copy = ${int(W_tile_size_nif * W_tile_size_nof * fs1 * fs2 * W_data_size_byte / 8)};
% endif
  dory_dma_memcpy_async(&DMA_copy_W);
  dory_dma_barrier(&DMA_copy_W);
% if ULTRA_VERBOSE:
  if (pi_core_id() == 0)
  {
    print_DMA_transfer(&DMA_copy_W, "Weight", l2_W, l1_buffer);
  }
% endif

% if flag_LUT:
  // popolate the LUT 
  if (pi_core_id() == 0)
  {
% if data_type_x[0] == "i":
    const int max_x_mag = (1 << (${x_data_size_byte} - 1));
% else:
    const int max_x_mag = (1 << ${x_data_size_byte}) - 1;
% endif
    const int num_in = max_x_mag + 1;

    const int max_w_mag = (1 << (${W_data_size_byte} - 1));
    const int num_w = max_w_mag + 1;

    int16_t *lut_buffer = (int32_t *)((uint8_t *)l1_buffer + ${buffer_l1_all});

    for (int x_mag = 0; x_mag < num_in; x_mag++)
    {
      for (int w_mag = 0; w_mag < num_w; w_mag++)
      {
        int32_t prod = (int32_t)x_mag * (int32_t)w_mag;
        lut_buffer[x_mag * num_w + w_mag] = (int16_t)prod;
      }
    }
  }
% endif

  pi_cl_team_barrier(0);

  // loop over the total number of tiles
% if flag_DW == 0:
  int total_tiles = ${tile_dim_nof * tile_dim_nif * tile_dim_h * tile_dim_w};
% else:
  int total_tiles = ${tile_dim_nof * tile_dim_h * tile_dim_w};
% endif
  for (iter = 0; iter < total_tiles; iter++) 
  {
    // update indices
% if tile_dim_nif != 1 and flag_DW == 0:
    // loop nest is nof,h,w,nif
    _i_nif_load += 1;
    if (_i_nif_load == ${tile_dim_nif})
    {
      _i_nif_load = 0;
% endif
      _i_w_load += 1;
      if (_i_w_load == ${tile_dim_w})
      {
        _i_w_load = 0;
        _i_h_load += 1;
        if (_i_h_load == ${tile_dim_h})
        {
          _i_h_load = 0;
% if flag_DW == 1:
          _i_nif_load += 1;
% endif
          _i_nof_load += 1;
        }
      }
% if tile_dim_nif != 1 and flag_DW == 0:
    }
% endif

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? ${x_tile_size_byte} : 0;
    db_W = !db_state_W ? ${W_tile_size_byte} : 0;
    db_y = !db_state_y ? ${y_tile_size_byte} : 0;
% if FLAG_BATCHNORM == 1:
    db_act = !db_state_W ? ${k_tile_size_byte_transfer} : 0;
% endif
% if tile_dim_nif * tile_dim_h * tile_dim_w != 1:
    exec_db_x = db_state_x ? ${x_tile_size_byte} : 0;
% else:
    exec_db_x = 0;
% endif
    db_state_x = !db_state_x;
    exec_db_W = db_state_W ? ${W_tile_size_byte} : 0;
% if FLAG_BATCHNORM == 1:
    exec_db_act = db_state_W ? ${k_tile_size_byte_transfer} : 0;
% endif
    if (_i_nif_load != _i_nif_exec || _i_nof_load != _i_nof_exec)
      db_state_W = !db_state_W;

    // switch all double buffering offset and y only after 
    // that all n_input_features have been analyzed: we need 
    // to pass all n_in to produce a single fil double buffered reads
    if (iter < (total_tiles - 1))
    {
      asm volatile("": : :"memory");
% if tile_dim_nif * tile_dim_h * tile_dim_w != 1:
      x_tile_size_nif = (_i_nif_load + 1 == ${tile_dim_nif}) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
      x_tile_size_h = (_i_h_load + 1 == ${tile_dim_h}) ? ${x_tile_size_h_last} : ${x_tile_size_h};
      x_tile_size_w = (_i_w_load + 1 == ${tile_dim_w}) ? ${x_tile_size_w_last} : ${x_tile_size_w};
      x_tile_size_byte = x_tile_size_nif * x_tile_size_h * x_tile_size_w * ${x_data_size_byte} / 8;
      x_length_nif_byte = (_i_nif_load + 1 == ${tile_dim_nif}) ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};
      
      // additionally overlap by padding for the first tile after a border one
      // this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      pad_offset_h = 0, pad_offset_w = 0;
      if (_i_h_load > 0)
        pad_offset_h = ${padding_top};
      if (_i_w_load > 0)
        pad_offset_w = ${padding_left};
% endif
      y_tile_size_h = (_i_h_load + 1 == ${tile_dim_h}) ? ${y_tile_size_h_last} : ${y_tile_size_h};
      y_tile_size_w = (_i_w_load + 1 == ${tile_dim_w}) ? ${y_tile_size_w_last} : ${y_tile_size_w};
      W_tile_size_nof = (_i_nof_load+1 == ${tile_dim_nof}) ? ${W_tile_size_nof_last} : ${W_tile_size_nof};
      W_tile_size_nif = (_i_nif_load+1 == ${tile_dim_nif}) ? ${W_tile_size_nif_last} : ${W_tile_size_nif};
% if flag_DW == 1:
      W_tile_size_byte = W_tile_size_nof * W_tile_size_nif * ${fs1} * ${fs2};
% else:
      W_tile_size_byte = W_tile_size_nof * W_tile_size_nif * ${W_data_size_byte} * ${fs1} * ${fs2} / 8;
% endif
      W_length_nif_byte = (_i_nif_load+1 == ${tile_dim_nif}) ? ${W_tile_size_nif_byte_last} : ${W_tile_nif_byte};
      
% if tile_dim_nif * tile_dim_h * tile_dim_w != 1:
      // prefetch next input
      DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif*g},  ${conv_overlap1}, ${conv_overlap2}, 0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte});
      DMA_copy_x.loc = (l1_buffer + ${l1_x_offset}) + db_x;
      DMA_copy_x.number_of_2d_copies = x_tile_size_h;
      DMA_copy_x.number_of_1d_copies = x_tile_size_w;
      DMA_copy_x.length_1d_copy = x_length_nif_byte;
      dory_dma_memcpy_async(&DMA_copy_x);
% endif
      // transfer of next weight tile if changed input or output channels
      if (_i_nif_load != _i_nif_exec || _i_nof_load != _i_nof_exec)
      {
% if flag_DW == 0:
        DMA_copy_W.ext = dory_get_tile_3d(l2_W, _i_nof_load, 0, _i_nif_load, ${W_tile_size_nof}, ${fs1}*${fs2}, ${W_tile_size_nif}, ${fs1}*${fs2}, ${nif}, 0, 0, 0, 0, 0, 0, ${W_data_size_byte});
% else:
        DMA_copy_W.ext = dory_get_tile_3d(l2_W, _i_nof_load, 0, 0, ${W_tile_size_nof}, ${fs1}*${fs2}, ${W_tile_size_nif}, ${fs1}*${fs2}, ${nif}, 0, 0, 0, 0, 0, 0, ${W_data_size_byte});
% endif
        DMA_copy_W.loc = (l1_buffer + ${l1_W_offset}) + db_W;
% if flag_DW == 0:
        DMA_copy_W.number_of_2d_copies = W_tile_size_nof;
        DMA_copy_W.length_1d_copy = W_length_nif_byte;
% else:
        DMA_copy_W.number_of_2d_copies = 1;
        DMA_copy_W.length_1d_copy = (int) W_tile_size_nof * ${W_data_size_byte} * ${ fs1 * fs2} / 8;
% endif
        dory_dma_memcpy_async(&DMA_copy_W);
% if FLAG_BATCHNORM == 1:
        // transfer BN parameters
        DMA_copy_k.ext = (uint32_t) l2_W + ${l2_off_k} + ${k_tile_size_byte_transfer} * _i_nof_load;
        DMA_copy_k.loc = (uint32_t) l1_buffer + ${l1_k_offset} + db_act;
        DMA_copy_k.length_1d_copy = (uint16_t) W_tile_size_nof * ${int(act_dim_bit/8)};
        dory_dma_memcpy_async(&DMA_copy_k);

        DMA_copy_lambda.ext = (uint32_t) l2_W + ${l2_off_lambda} + ${lambda_tile_size_byte_transfer} * _i_nof_load;
        DMA_copy_lambda.loc = (uint32_t) l1_buffer + ${l1_lambda_offset} + db_act;
        DMA_copy_lambda.length_1d_copy = (uint16_t) W_tile_size_nof * ${int(act_dim_bit/8)};
        dory_dma_memcpy_async(&DMA_copy_lambda);
% endif
      }
    }

    // creation of the pointers to input, output, weights, lambda and k
% if flag_DW == 1:
    asm volatile("": : :"memory");
% endif
    x = (${type} *)(l1_buffer + ${l1_x_offset} + exec_db_x);
% if FLAG_BATCHNORM == 1:
% if act_dim_bit == 32:
    k = (int32_t *)(l1_buffer + ${l1_k_offset} + exec_db_act);
    lambda = (int32_t *)(l1_buffer + ${l1_lambda_offset} + exec_db_act);
% else:
    k = (int64_t *)(l1_buffer + ${l1_k_offset} + exec_db_act);
    lambda = (int64_t *)(l1_buffer + ${l1_lambda_offset} + exec_db_act);
% endif
% endif
% if has_bias == 1:
    b = (${type} *)(l1_buffer + ${l1_b_offset} + _i_nof_exec*${bias_tile_size_byte});
% endif
    W = (${type} *)(l1_buffer + ${l1_W_offset} + exec_db_W);
    y = (${type} *)(l1_buffer + ${l1_y_offset} + db_y);
    // parameter passed to the kernel. Input and output sizes
    x_tile_size_nif_exec = (_i_nif_exec + 1 == ${tile_dim_nif}) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
    x_tile_size_h_exec = (_i_h_exec + 1 == ${tile_dim_h}) ? ${x_tile_size_h_last} : ${x_tile_size_h};
    x_tile_size_w_exec = (_i_w_exec + 1 == ${tile_dim_w}) ? ${x_tile_size_w_last} : ${x_tile_size_w};
    y_tile_size_nof = (_i_nof_exec + 1 == ${tile_dim_nof}) ? ${y_tile_size_nof_last} : ${y_tile_size_nof};
    y_tile_size_h = (_i_h_exec + 1 == ${tile_dim_h}) ? ${y_tile_size_h_last} : ${y_tile_size_h};
    y_tile_size_w = (_i_w_exec + 1 == ${tile_dim_w}) ? ${y_tile_size_w_last} : ${y_tile_size_w};
    y_tile_size_byte = y_tile_size_nof * y_tile_size_h * y_tile_size_w * ${y_data_size_byte} / 8;
    y_length_nof_byte = (_i_nof_exec + 1 == ${tile_dim_nof}) ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
    // padding
    p_r = 0;
    p_l = 0;
    p_t = 0;
    p_b = 0;
    if (_i_h_exec == 0)
      p_t = ${padding_top};
    if (_i_w_exec == 0)
      p_l = ${padding_left};
    if (_i_h_exec == ${tile_dim_h}-1)
      p_b = ${padding_bottom};
    if (_i_w_exec == ${tile_dim_w}-1)
      p_r = ${padding_right};

    // synch cluster
    pi_cl_team_barrier(0);
% if tile_dim_nof * tile_dim_nif * tile_dim_h * tile_dim_w == 1 or flag_DW == 1:
    // synch memory status
    asm volatile("": : :"memory");
% endif
    // kernel call
## pick the right function declaration
% if flag_DW == 0 and optional_type == '8bit' and (fs1 * fs2 > 1 or stride > 1):
    pulp_nn_conv_Ho_parallel(
% elif flag_DW == 0 and optional_type == '8bit' and fs1 * fs2 == 1  and 'FullyConnected' not in func_name:
    pulp_nn_pointwise_HoWo_parallel(
% elif flag_DW == 0 and optional_type == '8bit' and y_data_size_byte == 32 and ('FullyConnected' in func_name):
    pulp_nn_linear_out_32(
% elif flag_DW == 0 and optional_type == '8bit' and ('FullyConnected' in func_name):
    pulp_nn_linear(
% elif flag_DW == 0 and optional_type == 'mixed-hw' and conv1d:
    xpulp_nn_conv1d_${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}_${data_type_weights[0]}${W_data_size_byte}(
% elif flag_DW == 0 and 'mixed' in optional_type  and ('Conv' in func_name):
    ${"x" if 'hw' in optional_type else ""}pulp_nn_conv_${'lut_' if flag_LUT else ''}${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}_${data_type_weights[0]}${W_data_size_byte}(
% elif flag_DW == 0 and 'mixed' in optional_type  and ('Gemm' in func_name or 'MatMul' in func_name or 'FullyConnected' in func_name) and y_data_size_byte == 32:
    ${"x" if 'hw' in optional_type else ""}pulp_nn_linear_${'lut_' if flag_LUT else ''}${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}_${data_type_weights[0]}${W_data_size_byte}(
% elif flag_DW == 0 and 'mixed' in optional_type  and ('Gemm' in func_name or 'MatMul' in func_name or 'FullyConnected' in func_name):
    ${"x" if 'hw' in optional_type else ""}pulp_nn_linear_${'lut_' if flag_LUT else ''}${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}_${data_type_weights[0]}${W_data_size_byte}(
% elif flag_DW == 1 and optional_type == '8bit' and fs1 == 3 and fs2 == 3 and stride==1:
    pulp_nn_depthwise_generic(
% elif flag_DW == 1 and optional_type == '8bit' and fs1*fs2 < 4:
    pulp_nn_depthwise_generic_less_4_weights(
% elif flag_DW == 1 and optional_type == '8bit':
    pulp_nn_depthwise_generic(
% elif flag_DW == 1 and 'mixed' in optional_type:
    ${"x" if 'hw' in optional_type else ""}pulp_nn_depthwise_${'lut_' if flag_LUT else ''}${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}_${data_type_weights[0]}${W_data_size_byte}(
% endif
## passing the right arguments
% if 'Gemm' in func_name or 'FullyConnected' in func_name:
## Dense layer
% if has_bias:
      x, 
      b, 
      y, 
      W,
% else:
      x, 
      0, 
      y, 
      W,
% endif
% if flag_LUT:
      lut,
% endif
% if FLAG_BATCHNORM == 1 and y_data_size_byte != 32:
      k, 
      lambda,
% elif y_data_size_byte != 32:
      0, 
      0,
% endif
% if y_data_size_byte != 32:
% if FLAG_RELU == 1:
      out_mult, 
      out_shift,
% else:
      1, 
      out_shift,
% endif
% endif
      x_tile_size_nif_exec, 
      y_tile_size_nof${"," if y_data_size_byte != 32 else ""}
% if y_data_size_byte != 32:
      ${FLAG_RELU}, 
      ${FLAG_BATCHNORM}
% endif
      );
% else:
## conv-like layers
      x, 
      im2col,
% if has_bias:
      b,
% else:
      NULL,
% endif
      y, 
      W,
% if flag_LUT:
      lut,
% endif
% if flag_DW == 1:
      pwt_buffer,
% endif
% if FLAG_BATCHNORM == 1:
      k, 
      lambda,
% else:
      0, 
      0,
% endif
      out_mult, 
      out_shift,
      x_tile_size_w_exec${", x_tile_size_h_exec" if not conv1d else ""}, 
      x_tile_size_nif_exec,
      y_tile_size_w${", y_tile_size_h" if not conv1d else ""}, 
      y_tile_size_nof,
      ${fs2},${f"{ fs1}," if not conv1d else ""}
      ${f"p_t, p_b, " if not conv1d else ""} 
      p_l, 
      p_r, 
      ${stride}${f", {stride}" if not conv1d else ""},
      ${f"{dilations[1]}," if conv1d else ""}
      ${FLAG_RELU}, 
      ${FLAG_BATCHNORM}
    );
% endif

    // synch cluster
    pi_cl_team_barrier(0);

% if tile_dim_nif != 1 and flag_DW == 0:
    if (_i_nif_load == 0)
    {
% endif
      // wait until the previous output is not stored in L2
      if (iter > 0)
        dory_dma_barrier(&DMA_copy_y);
      // wait until the pre-fetched input and weight are not ready in L1
% if tile_dim_nif * tile_dim_h * tile_dim_w != 1:
      dory_dma_barrier(&DMA_copy_x);
% endif;
      dory_dma_barrier(&DMA_copy_W);
% if ULTRA_VERBOSE:
      if (pi_core_id() == 0)
      {
        if (iter > 0)
        {
          print_DMA_transfer(&DMA_copy_y, "Output", l2_y, l1_buffer);
          debug_print_tensor(&DMA_copy_y, "Output L2", 512);
        }
% if tile_dim_nif * tile_dim_h * tile_dim_w != 1:
        print_DMA_transfer(&DMA_copy_x, "Input", l2_x, l1_buffer);
% endif
        print_DMA_transfer(&DMA_copy_W, "Weight", l2_W, l1_buffer);
      }
      
% endif
% if FLAG_BATCHNORM == 1:
      if (iter < (total_tiles - 1) && (_i_nif_load != _i_nif_exec || _i_nof_load != _i_nof_exec))
      {
        dory_dma_barrier(&DMA_copy_k);
        dory_dma_barrier(&DMA_copy_lambda);
      }
% endif
      // write back the output tile
      DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${int(nof*factor)}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
      DMA_copy_y.loc = (l1_buffer + ${l1_y_offset}) + db_y;
      DMA_copy_y.number_of_2d_copies = y_tile_size_h;
      DMA_copy_y.number_of_1d_copies = y_tile_size_w;
      DMA_copy_y.length_1d_copy = y_length_nof_byte;
      
% if tile_dim_nif != 1 and flag_DW == 0:
    }
% endif
    dory_dma_memcpy_async(&DMA_copy_y);

    // update prev iterators
    db_state_y = !db_state_y;
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
    pi_cl_team_barrier(0);
  }


% if not TEST:
  // wait for final write and clean memory
  dory_dma_barrier(&DMA_copy_y);
% if ULTRA_VERBOSE:
  if (pi_core_id() == 0)
  {
    print_DMA_transfer(&DMA_copy_y, "Output", l2_y, l1_buffer);
    debug_print_tensor(&DMA_copy_y, "Output L2", 512);
  }
% endif
  dory_dma_free(&DMA_copy_y);
% endif
}
