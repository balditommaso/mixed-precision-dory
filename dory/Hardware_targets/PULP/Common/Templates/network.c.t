
<%
l3_supported = DORY_HW_graph[0].HW_description['memory']['levels'] > 2
%>\
#define DEFINE_CONSTANTS
%if not l3_supported:
#include "${prefix}weights.h"
%endif
#include "net_utils.h"
#include "pmsis.h"
#include "${prefix}network.h"
#include "directional_allocator.h"
#include "mem.h"
#include <string.h>
% for layer in list_h:
#include "${layer}"
% endfor

% if sdk == 'pulp-sdk':
#define ICACHE_CTRL_UNIT 0x10201400
#define ICACHE_PREFETCH ICACHE_CTRL_UNIT + 0x1C
% endif

% if verbose:
#define VERBOSE 1
% endif

// CONSTANTS
% if l3_supported:
#define L3_WEIGHTS_SIZE 4000000
#define L3_INPUT_SIZE 1500000
#define L3_OUTPUT_SIZE 1500000
% endif
static void *L3_weights = NULL;
static void *L3_input = NULL;
static void *L3_output = NULL;

% if 'Yes' in performance or 'Perf_final' in verbose_level:
int ${prefix}cycle_network_execution;
% endif
% if l3_supported:
/* Moves the weights and the biases from hyperflash to hyperram */
void ${prefix}network_initialize() 
{
  L3_weights = ram_malloc(L3_WEIGHTS_SIZE);
  L3_input = ram_malloc(L3_INPUT_SIZE);
  L3_output = ram_malloc(L3_OUTPUT_SIZE);

#ifdef VERBOSE
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_weights, L3_weights?"Ok":"Failed");
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_input, L3_input?"Ok":"Failed");
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_output, L3_output?"Ok":"Failed");
#endif

  void *w_ptr = L3_weights;
  for (int i = 0; i < ${weights_number}; i++) 
  {
    size_t size = load_file_to_ram(w_ptr, L3_weights_files[i]);
    L3_weights_size[i] = size;
    w_ptr += size;
  }
}
% endif

% if l3_supported:
/* Remove RAM memory */
void ${prefix}network_terminate() 
{
  ram_free(L3_weights, L3_WEIGHTS_SIZE);
  ram_free(L3_input, L3_INPUT_SIZE);
  ram_free(L3_output, L3_OUTPUT_SIZE);
}
% endif

void ${prefix}execute_layer_fork(void *args) 
{
  layer_args_t *layer_args = (layer_args_t *)args;
  if (pi_core_id() == 0) 
    layer_args->L1_buffer = pmsis_l1_malloc(${l1_buffer});

  switch (layer_args->layer_id)
  {
% for i in range(len(DORY_HW_graph)):
    case ${i}:
      pi_cl_team_fork(NUM_CORES, (void *)${func_name[i]}, args);
      break;
% endfor
  }

  if (pi_core_id() == 0) pmsis_l1_malloc_free(layer_args->L1_buffer, ${l1_buffer});
}

struct ${prefix}network_run_token ${prefix}network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir${", void *L2_input_h" if not l3_supported else ""})
{
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;
<%
    n_args = 5 if l3_supported else 6
%>\
  unsigned int args[${n_args}];
  args[0] = (unsigned int) l2_buffer;
  args[1] = (unsigned int) l2_buffer_size;
  args[2] = (unsigned int) l2_final_output;
  args[3] = (unsigned int) exec;
  args[4] = (unsigned int) initial_dir;
  % if not l3_supported:
  args[5] = (unsigned int) L2_input_h;
  % endif
  // open cluster...
  pi_cluster_task(&cluster_task, ${prefix}network_run_cluster, args);
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return;
  // Then offload an entry point, this will get executed on the cluster controller
  cluster_task.stack_size = ${master_stack};
  cluster_task.slave_stack_size = ${slave_stack};
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  return (struct ${prefix}network_run_token) {
    .cluster_dev = cluster_dev
  };
}

void ${prefix}network_run_wait(struct ${prefix}network_run_token token)
{
  pi_cluster_close(&token.cluster_dev);
  % if 'Perf_final' in verbose_level:
  print_perf("Final", ${prefix}cycle_network_execution, ${MACs});
  % endif
}

void ${prefix}network_run(void *l2_buffer, 
                          size_t l2_buffer_size, 
                          void *l2_final_output, 
                          int exec, 
                          int initial_dir${", \nvoid *L2_input_h" if not l3_supported else ""})
{
  ${prefix}network_run_wait(network_run_async(l2_buffer, l2_buffer_size, l2_final_output, exec, initial_dir${", L2_input_h" if not l3_supported else ""}));
}

void ${prefix}network_run_cluster(void *args) 
{
  // un-pack the arguments
  unsigned int * real_args = (unsigned int *) args;
  void * l2_buffer = (void *) real_args[0];
  size_t l2_buffer_size = (size_t) real_args[1];
  void * l2_final_output = (void *) real_args[2];
  int exec = (int) real_args[3];
  int dir = (int) real_args[4];
% if not l3_supported:
  void * L2_input_h = (void *)real_args[5];
% endif
  // prepare pointers to memories
  void *L2_output = NULL;
  void *L2_input = NULL;
  void *L2_weights = NULL;
  void *L3_weights_curr = L3_weights;
  void *bypass_activations = NULL;
  // local variables 
  int residual_number = 0;
  int bypass_dimension = 0;
% if not l3_supported:
  int left_branch_nodes = 0, right_branch_nodes = 0;
  int z = 0;
  int end_left = 0;
% endif
  int perf_cyc = 0;
% if not l3_supported:
  L2_input = L2_input_h;
% endif

  // set the pointers of the allocator
  directional_allocator_init(l2_buffer, l2_buffer_size);

% if 'Yes' in performance or 'Perf_final' in verbose_level:
  // perf measurement begin
  ${prefix}cycle_network_execution = 0;
% endif

  // count how many layers with weights we have processed to increment the weights_L3 pointer
  int weight_l_cnt = 0; 
  for (int i = 0; i < ${len(DORY_HW_graph)}; i++) 
  {
    // allocate speace for the output
    L2_output = dmalloc(activations_out_size[i], !dir);

% if l3_supported:
    // if necessary reserve space for input from L3
    if (L3_input_layers[i] == 1)
      L2_input = dmalloc(activations_size[i], dir);
    // if necessary reserve space for weights from L3
    if (layer_with_weights[i] == 1)
      L2_weights = dmalloc(weights_size[i], dir);
    // allocate the weights
    if (allocate_layer[i] == 1)
      cl_ram_read(L2_weights, L3_weights_curr, weights_size[i]);
% else:
    L2_weights = Weights_name[i];
% endif
% if 'Check_all' in verbose_level and render_checksum:
#ifdef VERBOSE
% if l3_supported:
    if (L3_input_layers[i] == 1)
      printf("Input in L3\n");
    else
% endif
    if (i == 0 || branch_change[i-1] == 0) {
      checksum("L2 input", L2_input, activations_size[i], activations_checksum[i][exec]);
% if l3_supported:
      if (allocate_layer[i] == 1)
% else:
      if (layer_with_weights[i])
% endif
        checksum("L2 weights", L2_weights, weights_size[i], weights_checksum[i]);
      else
        printf("Weights in L3\n");
    }
    else
      printf("Switching branch, already checked activation\n");
#endif
% endif
    // struct with pointers for the executor
    layer_args_t largs = {
      .L3_input = (unsigned int) L3_input,
      .L3_output = (unsigned int) L3_output,
      .L3_after_weights = (unsigned int) L3_weights_curr,
      .L2_input = (unsigned int) L2_input,
      .bypass = (unsigned int) bypass_activations,
      .L2_output = (unsigned int) L2_output,
      .L2_weights = (unsigned int) L2_weights,
      .L1_buffer = 0,
      .ram = (unsigned int) get_ram_ptr(),
      .out_mult = (unsigned int) out_mult_vector[i],
      .out_shift = (unsigned int) out_shift_vector[i],
      .layer_id = i
    };
% if 'Yes' in performance or 'Perf_final' in verbose_level:
    // BEGIN perf measurement
    pi_perf_conf(1 << PI_PERF_CYCLES);
    pi_perf_reset();
    pi_perf_stop();
    pi_perf_start();
% endif

    // layer execution
    ${prefix}execute_layer_fork((void *) &largs);

% if 'Yes' in performance or 'Perf_final' in verbose_level:
    // END performance measurements
    pi_perf_stop();
    perf_cyc =  pi_perf_read(PI_PERF_CYCLES);
    ${prefix}cycle_network_execution += perf_cyc;
% endif

% if 'Yes' in performance:
    print_perf(Layers_name[i], perf_cyc, NODEs_MACS[i]);
    log_perf_csv(Layers_name[i], perf_cyc, NODEs_MACS[i]);
% endif

    // 3-way swap: the old output is the new input
    asm volatile("": : :"memory");
    unsigned int temp = L3_input;
    L3_input = L3_output;
    asm volatile("": : :"memory");
    L3_output = temp;
    asm volatile("": : :"memory");

#ifdef VERBOSE
    printf("Layer %s %d ended: \n", Layers_name[i], i);
% if 'Check_all' in verbose_level and render_checksum:
% if l3_supported:
    if (L3_output_layers[i] == 1) 
    {
      printf("Output in L3. Expected checksum: %d\n", activations_out_checksum[i][exec]);
    } 
    else 
    {
% endif
      checksum(i + 1 < ${len(DORY_HW_graph)} ? "L2 output" : "final output", L2_output, activations_out_size[i], activations_out_checksum[i][exec]);
% if l3_supported:
    }
% endif
    printf("\n");
% elif 'Last' in verbose_level:
    if (i == ${len(DORY_HW_graph) - 1})
        checksum("final layer", L2_output, activations_out_size[i], activations_out_checksum[i][exec]);
% endif
#endif

    // free memory
% if l3_supported:
    if (layer_with_weights[i] == 1)
      dfree(weights_size[i], dir);
    dfree(activations_size[i], dir);
% endif
    if (branch_input[i] == 1)
      dfree(bypass_dimension, dir);
    L2_input = L2_output;
% if not l3_supported:
    if  (branch_output[i]==1)
      {
        bypass_activations = L2_output;
        bypass_dimension = activations_out_size[i];
      }

    if (i > 0 && branch_output[i-1] == 0 && branch_change[i-1] == 0)
      dfree(activations_size[i], dir);
% endif

    // Residual connections
    if (i < ${len(DORY_HW_graph) - 1}) 
    {
% if l3_supported:
      if (branch_input[i+1] == 1) 
      {
        bypass_activations = dmalloc(bypass_dimension, !dir);
        residual_number--;
        cl_ram_read(bypass_activations, layers_pointers[residual_number], bypass_dimension);
        cl_ram_free(layers_pointers[residual_number], bypass_dimension);
      }

      if (i > 0 && branch_output[i-1] == 1 && L3_input_layers[i] == 1) 
      { 
        L3_input = cl_ram_malloc(L3_INPUT_SIZE);
      }
      if (branch_output[i] == 1 && L3_output_layers[i] == 1) 
      {
        cl_ram_free(L3_input + activations_out_size[i], L3_INPUT_SIZE - activations_out_size[i]);
        layers_pointers[residual_number] = L3_input;
        residual_number++;
        bypass_dimension = activations_out_size[i];
      } 
      else if (branch_output[i] == 1 || branch_change[i] == 1) 
      {
        layers_pointers[residual_number] = cl_ram_malloc(activations_out_size[i]);
        cl_ram_write(layers_pointers[residual_number], L2_output, activations_out_size[i]);
        residual_number++;
        bypass_dimension = activations_out_size[i];
      }

      if (branch_change[i] == 1) 
      {
        dfree(activations_out_size[i], !dir);
        L2_input = dmalloc(activations_size[i + 1], !dir);
        cl_ram_read(L2_input, layers_pointers[residual_number - 2], activations_size[i + 1]);
        cl_ram_free(layers_pointers[residual_number - 2], activations_size[i + 1]);
      }
      if (L3_output_layers[i] == 1)
        dfree(activations_out_size[i], !dir);
% else:
      if  (branch_output[i] == 1)
      {
        left_branch_nodes = 0;
        right_branch_nodes = 0;
        z = i + 1;
        end_left = 0;
        while (branch_input[z] == 0)
        {
          if (end_left == 0)
            left_branch_nodes += 1;
          else
            right_branch_nodes += 1;
          if (branch_change[z] == 1)
            end_left = 1;
          z += 1;
        }
        if ((left_branch_nodes % 2 == 1) && (right_branch_nodes == 0))
          dir = !dir;
        if ((left_branch_nodes % 2 == 0) && (right_branch_nodes > 0))
          dir = !dir;
      }

      if (branch_change[i]==1)
      {
        L2_input = bypass_activations;
        bypass_activations = L2_output;
        bypass_dimension = activations_out_size[i];
        if (right_branch_nodes % 2 == 1)
          dir = !dir;
      }
% endif
    }
% if l3_supported:
    // shift the pointer to the weights of the next layer
    if (layer_with_weights[i])
       L3_weights_curr += L3_weights_size[weight_l_cnt++];
% endif
    // change direction of the double buffer
    dir = !dir;
  }
  // copy the network output back to L2
  for (int i = 0; i < activations_out_size[${len(DORY_HW_graph)-1}]; i++)
  {
    *((uint8_t*)(l2_final_output + i)) = *((uint8_t*)(L2_output + i));
  }
  
  // Copy the final network output into the caller-provided L2 buffer.
  memcpy(
      l2_final_output,
      L2_output,
      activations_out_size[${len(DORY_HW_graph) - 1}]
  );

<%
final_node = DORY_HW_graph[-1]

final_bits = int(final_node.output_activation_bits)

final_type = getattr(
    final_node,
    "output_activation_type",
    "int",
)
final_signed = final_type == "int"

final_channels = int(final_node.output_channels)
final_h = int(final_node.output_dimensions[0])
final_w = int(final_node.output_dimensions[1])
num_outputs = final_channels * final_h * final_w
%>\

  printf("\nFinal output:\n");

% if final_bits == 2:
  {
    const uint8_t *packed_output =
        (const uint8_t *)l2_final_output;

    for (int i = 0; i < ${num_outputs}; i++)
    {
      const int byte_index = i >> 2;
      const int shift = (i & 3) * 2;
      const uint8_t raw =
          (packed_output[byte_index] >> shift) & 0x3;

% if final_signed:
      const int decoded =
          (raw & 0x2) ? ((int)raw - 4) : (int)raw;
% else:
      const int decoded = (int)raw;
% endif

      printf("%d ", decoded);
    }
  }

% elif final_bits == 4:
  {
    const uint8_t *packed_output =
        (const uint8_t *)l2_final_output;

    for (int i = 0; i < ${num_outputs}; i++)
    {
      const int byte_index = i >> 1;
      const int shift = (i & 1) * 4;
      const uint8_t raw =
          (packed_output[byte_index] >> shift) & 0xF;

% if final_signed:
      const int decoded =
          (raw & 0x8) ? ((int)raw - 16) : (int)raw;
% else:
      const int decoded = (int)raw;
% endif

      printf("%d ", decoded);
    }
  }

% elif final_bits == 8:
  {
    const ${"int8_t" if final_signed else "uint8_t"} *final_output =
        (const ${"int8_t" if final_signed else "uint8_t"} *)
            l2_final_output;

    for (int i = 0; i < ${num_outputs}; i++)
    {
      printf("%d ", (int)final_output[i]);
    }
  }

% elif final_bits == 16:
  {
    const ${"int16_t" if final_signed else "uint16_t"} *final_output =
        (const ${"int16_t" if final_signed else "uint16_t"} *)
            l2_final_output;

    for (int i = 0; i < ${num_outputs}; i++)
    {
      printf("%d ", (int)final_output[i]);
    }
  }

% else:
#error "Unsupported final output precision"
% endif

  printf("\n");
}