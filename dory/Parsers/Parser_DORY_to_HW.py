

# Libraries
import numpy as np
import sys
import copy

# DORY modules
from .HW_node import HW_node
from dory.Utils.DORY_utils import Printer
from dory.Parsers.DORY_node import DORY_node
from dory.Hardware_targets.PULP.Common.Tiler import Tiler_PULP
from typing import *


class Parser_DORY_to_HW:
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(
        self, 
        graph: List[DORY_node], 
        rules: Dict[str, Any], 
        Pattern_rewriter, 
        supported_nodes: List[str], 
        HW_description: Dict[str, Any], 
        config_file: Dict[str, Any], 
        Tiler: Tiler_PULP, 
        network_directory: str = None, 
        n_inputs: int = 1,
        verify_checksum: bool = True,
        num_cores: int = 8
    ):
        self.supported_nodes = supported_nodes
        self.DORY_Graph = graph
        self.Printer_Frontend = Printer("logs/HW_related")
        self.Pattern_rewriter = Pattern_rewriter
        self.rules = rules
        self.HW_description = HW_description
        self.network_directory = network_directory
        self.config_file = config_file
        self.n_inputs = n_inputs
        self.verify_checksum = verify_checksum
        self.num_cores = num_cores
        HW_node.Tiler = Tiler


    def mapping_to_HW_nodes(self):
        print("\nBackend: Matching patterns from generated DORY ONNX to HW Nodes.")
        for i, node in enumerate(self.DORY_Graph):
            string_matching, indexes = self.pattern_matching(node, i)
            if isinstance(string_matching, str):
                self.DORY_Graph = self.Pattern_rewriter(
                    self.DORY_Graph
                ).execute(string_matching, indexes)


    def check_graph(self):
        for node in self.DORY_Graph:
            if node.name not in self.supported_nodes:
                sys.exit("\nDORY Backend Check. Node {} is not accepted inside the HW Frontend IR.\n".format(node.name))
        print("\nDORY checking of the graph: OK\n")


    def check_parameters(self):
        print("\nTo be implemented in the target backend")


    def pattern_matching(self, input_node, input_index):
        number_of_nodes = 0
        rule_found = False
        DORY_node_indexes_to_export = []
        for key, rule in self.rules.items():
            DORY_node_indexes = []
            DORY_node_indexes.append(input_index)
            if rule["number_of_nodes"] == 1 and input_node.name in rule["nodes_name"]:
                if number_of_nodes < rule["number_of_nodes"]:
                    rule_found = key
                    number_of_nodes = rule["number_of_nodes"]
                    DORY_node_indexes_to_export = DORY_node_indexes
            elif input_node.name in rule["nodes_name"]:
                node = input_node
                match = 1
                nodes = copy.deepcopy(rule["nodes_name"])
                index = nodes.index(node.name)
                nodes[index] = "Match"
                while match == 1:
                    match = 0
                    inputs = rule["dependencies"][str(index)]["inputs"]
                    outputs = rule["dependencies"][str(index)]["outputs"]
                    for nodes_index in inputs:
                        int_index = node.input_indexes
                        node_to_search = nodes[int(nodes_index)]
                        for i,node_i in enumerate(self.DORY_Graph):
                            if node_i.output_index in int_index and node_i.name == node_to_search:
                                nodes[int(nodes_index)] = "Match"
                                match = 1
                                DORY_node_indexes.append(i)
                    for nodes_index in outputs:
                        out_index = node.output_index
                        node_to_search = nodes[int(nodes_index)]
                        for i,node_i in enumerate(self.DORY_Graph):
                            if out_index in node_i.input_indexes and node_i.name == node_to_search:
                                nodes[int(nodes_index)] = "Match"
                                match = 1
                                DORY_node_indexes.append(i)
                                node = node_i
                                index = int(nodes_index)
                if sum(x=="Match" for x in nodes) == len(nodes):
                    if number_of_nodes < rule["number_of_nodes"]:
                        rule_found = key
                        number_of_nodes = rule["number_of_nodes"]
                        DORY_node_indexes_to_export = DORY_node_indexes
        return rule_found, DORY_node_indexes_to_export


    def update_branches_graph(self):
        print("\nDORY generic Frontend. Updating branches pointers.")
        
        graph = self.DORY_Graph
        
        for node in graph:
            node.add_existing_parameter(
                "branch_in", 
                1 if len(node.input_indexes) > 1 else 0
            )
            
            num_consumers = 0
            
            for consumer in graph:
                if node.output_index in consumer.input_indexes:
                    num_consumers += 1
                    
            node.add_existing_parameter(
                "branch_out",
                1 if num_consumers > 1 else 0
            )
            node.add_existing_parameter("branch_change", 0)
            node.add_existing_parameter("branch_last", 0)
            
        producers = {
            node.output_index: (index, node)
            for index, node in enumerate(graph)
        }
        
        for merge_node in graph:

            if merge_node.branch_in != 1:
                continue

            input_producers = []

            for input_index in merge_node.input_indexes:
                if input_index not in producers:
                    continue

                producer_index, producer_node = producers[input_index]
                input_producers.append(
                    (producer_index, producer_node)
                )

            if len(input_producers) != 2:
                continue

            input_producers.sort(key=lambda x: x[0])

            _, early_producer = input_producers[0]
            _, late_producer = input_producers[1]

            if (
                early_producer.branch_out != 1
                and late_producer.branch_out != 1
            ):
                early_producer.add_existing_parameter(
                    "branch_change", 1
                )
                late_producer.add_existing_parameter(
                    "branch_last", 1
                )
            else:
                early_producer.add_existing_parameter(
                    "branch_last", 1
                )


    def update_dimensions_graph(self):
        print("\nUpdating dimensions of vectors inside the graph, if they do not match among nodes")
        for i, node in enumerate(self.DORY_Graph):
            if i > 0:
                if isinstance(self.DORY_Graph[i].input_channels, type(None)):
                    if "FullyConnected" in self.DORY_Graph[i].name:
                        self.DORY_Graph[i].input_channels = int(
                            self.DORY_Graph[i-1].output_channels * \
                            np.prod(self.DORY_Graph[i-1].output_dimensions)
                        )
                    else:
                        self.DORY_Graph[i].input_channels = self.DORY_Graph[i-1].output_channels
                if len(self.DORY_Graph[i].input_dimensions) == 0:
                    self.DORY_Graph[i].input_dimensions = self.DORY_Graph[i-1].output_dimensions


    def add_tensors_memory_occupation_and_MACs(self):
        print("\nUpdating memory occupation and MACs of tensors in layers")
        for i, node in enumerate(self.DORY_Graph):
            if "Convolution" in node.name or "FullyConnected" in node.name or "Add" in node.op_type or "Pooling" in node.name:
                node.add_memory_and_MACs()


    def adjust_data_layout(self):
        print("\nTo be implemented in the target backend")


    # Override if you want to instanciate a different type of HW_node
    def transform_nodes_to_hw_nodes(self):
        self.DORY_Graph = [HW_node(node, self.HW_description) for node in self.DORY_Graph]


    def tiling(self):
        print("\nInsert tiling parameters per layer inside graph nodes")

        producer_map = {
            str(node.output_index): node
            for node in self.DORY_Graph
        }

        for node in self.DORY_Graph:
            input_nodes = []

            for input_index in node.input_indexes:
                producer = producer_map.get(str(input_index))

                if producer is not None:
                    input_nodes.append(producer)

            primary_node = (
                input_nodes[0]
                if input_nodes
                else node
            )

            node.create_tiling_dimensions(
                primary_node,
                self.config_file,
                num_cores=self.num_cores,
                input_nodes=input_nodes,
            )
        

    def renaming_weights(self):
        print("\nDORY Backend: Renaming Weights tensors.")
        for i, node in enumerate(self.DORY_Graph):            
            node.rename_weights()           
            
            
    def reorder_graph_branch_contiguous(self):
        """
        Reorder self.DORY_Graph in a valid topological execution order such
        that, at a fork, one branch is completed before starting the sibling.

        The shorter branch is executed first. This matches the assumptions
        behind DORY's branch_change / branch_last runtime bookkeeping.
        """

        graph = self.DORY_Graph
        n = len(graph)

        if n <= 1:
            return

        producer_of = {
            str(node.output_index): i
            for i, node in enumerate(graph)
        }

        predecessors = {i: set() for i in range(n)}
        successors = {i: set() for i in range(n)}

        for i, node in enumerate(graph):
            for input_index in node.input_indexes:
                producer = producer_of.get(
                    str(input_index)
                )

                if producer is None:
                    continue

                predecessors[i].add(producer)
                successors[producer].add(i)

        remaining_successors = {
            i: len(successors[i])
            for i in range(n)
        }

        depth = {}
        stack = [i for i in range(n) if remaining_successors[i] == 0]

        for i in stack:
            depth[i] = 0

        while stack:
            current = stack.pop()

            for predecessor in predecessors[current]:
                remaining_successors[predecessor] -= 1

                if remaining_successors[predecessor] == 0:
                    depth[predecessor] = 1 + max(
                        (
                            depth[successor]
                            for successor in successors[predecessor]
                        ),
                        default=0,
                    )
                    stack.append(predecessor)

        if len(depth) != n:
            raise ValueError(
                "Cannot compute DORY execution order: "
                "graph contains a cycle or invalid dependencies."
            )

        in_degree = {i: len(predecessors[i]) for i in range(n)}

        ready = [i for i in range(n) if in_degree[i] == 0]

        ready.sort(key=lambda i: (depth[i], i), reverse=True)

        order = []

        while ready:
            current = ready.pop()
            order.append(current)

            newly_ready = []

            for successor in successors[current]:
                in_degree[successor] -= 1

                if in_degree[successor] == 0:
                    newly_ready.append(successor)

            newly_ready.sort(
                key=lambda i: (depth[i], i),
                reverse=True,
            )

            ready.extend(newly_ready)

        if len(order) != n:
            raise ValueError(
                "Could not generate complete branch-contiguous "
                "DORY execution order."
            )

        self.DORY_Graph = [graph[i] for i in order]


    def formatting_constant_parameters_tensors_and_activations(self):
        print("\nDORY Backend: Formatting constants and adding checksums")
        # for i, node in enumerate(self.DORY_Graph):            
        #     node.add_checksum_w_integer()           
        #     if self.verify_checksum and self.network_directory is not None:
        #         node.add_checksum_activations_integer(self.network_directory, i, self.n_inputs)


    def full_graph_parsing(self):
        print("#####################################################")
        print("## DORY GENERAL PARSING FROM DORY IR TO DORY HW IR ##")
        print("## FINAL RAPRESENTATION: DORY HW IR                ##")
        print("#####################################################")
        self.Printer_Frontend.print_json_from_DORY_graph("00_DORY_HW_input_graph", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("00_DORY_HW_input_graph", self.DORY_Graph)
        self.reorder_graph_branch_contiguous()
        self.mapping_to_HW_nodes()
        self.Printer_Frontend.print_json_from_DORY_graph("01_DORY_HW_graph_raw", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("01_DORY_HW_graph_raw", self.DORY_Graph)
        self.update_branches_graph()
        self.Printer_Frontend.print_json_from_DORY_graph("02_DORY_HW_graph_fixed_branches", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("02_DORY_HW_graph_fixed_branches", self.DORY_Graph)
        self.update_dimensions_graph()
        self.Printer_Frontend.print_json_from_DORY_graph("03_DORY_HW_graph_fixed_dimensions", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("03_DORY_HW_graph_fixed_dimensions", self.DORY_Graph)
        self.adjust_data_layout()
        self.Printer_Frontend.print_json_from_DORY_graph("04_DORY_HW_adjusted_data_layout", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("04_DORY_HW_adjusted_data_layout", self.DORY_Graph)
        self.add_tensors_memory_occupation_and_MACs()
        self.Printer_Frontend.print_json_from_DORY_graph("05_DORY_HW_graph_added_tensors_dim", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("05_DORY_HW_graph_added_tensors_dim", self.DORY_Graph)
        self.transform_nodes_to_hw_nodes()
        self.tiling()
        self.Printer_Frontend.print_json_from_DORY_graph("06_DORY_HW_tiled_graph", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("06_DORY_HW_tiled_graph", self.DORY_Graph)
        self.renaming_weights()
        self.formatting_constant_parameters_tensors_and_activations() 
        self.Printer_Frontend.print_json_from_DORY_graph("07_DORY_HW_with_checksums", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("07_DORY_HW_with_checksums", self.DORY_Graph)
        self.check_graph()
        self.check_parameters()
        return self.DORY_Graph