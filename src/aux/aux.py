
import heapq


def UpdateProbabilityVector(counts, SelectedAmount):
    # Get the top 4 most probable candidates
    candidates = heapq.nlargest(SelectedAmount, counts, key=counts.get)

    # Get the number of qubits from the bitstring length (assuming all bitstrings are of the same length)
    n = len(candidates[0])  # Assuming all candidates are of the same length

    # Initialize ST as a zero vector of length n (number of qubits)
    ST = [0] * n

    # Concatenate all top candidates into a single string
    all_candidates_str = ''.join(candidates)

    # Accumulate the sums for each bit position across the concatenated candidates
    for i in range(len(all_candidates_str)):
        ST[i % n] += int(all_candidates_str[i])  # Convert binary character to in   teger

    # Normalize ST by the total number of selected candidates
    ST = [x / float(SelectedAmount) for x in ST]

    return ST




def UpdateProbabilityVectorCostFunction(counts, selectedAmount, maxAmount, observable, n):

    top_candidates = heapq.nlargest(maxAmount, counts, key=counts.get)
    evaluated = []
    for bitstr in top_candidates:
        state_int = int(bitstr, 2)
        cost = calculate_cost(state_int, observable)
        evaluated.append((bitstr, cost))

    candidates = heapq.nsmallest(selectedAmount, evaluated, key=lambda x: x[1])
    # Get the number of qubits from the bitstring length (assuming all bitstrings are of the same length)

    # Initialize ST as a zero vector of length n (number of qubits)
    ST = [0] * n

    # Concatenate all top candidates into a single string
    all_candidates_str = ''.join([bitstr for bitstr, _ in candidates])

    # Accumulate the sums for each bit position across the concatenated candidates
    for i in range(len(all_candidates_str)):
        ST[i % n] += int(all_candidates_str[i])  # Convert binary character to in   teger

    # Normalize ST by the total number of selected candidates
    ST = [x / float(selectedAmount) for x in ST]

    return ST



import os
import pickle

def read_pickle_files_from_directory(directory):

    loaded_files = {}
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return loaded_files

    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'rb') as file:
                    loaded_files[filename[:-4]] = pickle.load(file)
            except Exception as e:
                print(f"Error reading file '{filename}': {e}")


    return loaded_files

def get_graphs_name(loaded_data):
    return list(loaded_data.keys())


def number_qubits(graph_name):
    parts = graph_name.split('_')
    return int(parts[0])



import json
def read_result_data(graphs_str, name_str, folder_path):
    result_data = []
    for graph in graphs_str:
        # Build filename using format "graph_name: name_str.json"
        file_name = f"{graph}: {name_str}.json"
        file_path = os.path.join(folder_path, file_name)

        try:
            with open(file_path, "r") as f:
                result_data.append(json.load(f))
        except FileNotFoundError:
            print(f"[Warning] File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"[Error] Failed to parse JSON: {file_path}")

    return result_data







import rustworkx as rx

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        #paulis = ["I"] * 127
        #print(edge)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1])

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list





import numpy as np
from qiskit.quantum_info import SparsePauliOp

_PARITY = np.array([-1 if bin(i).count("1") % 2 else 1 for i in range(256)], dtype=np.complex128)


## Auxiliary function helping to calculate the cost for a given state (sample output)

def calculate_cost(state: int, observable: SparsePauliOp) -> complex:
    """Utility for the evaluation of the expectation value of a measured state."""
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8)
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced]).real


def best_sample(samples, hamiltonian):
    """Find solution with lowest cost"""
    min_sol = None

    min_cost = float("inf")

    for sample in samples.keys():
        cost = calculate_cost(int(sample), hamiltonian)
        if min_cost > cost:
            min_cost = cost
            min_sol = sample

    return min_sol

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

from typing import Sequence
def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(list(graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
    return sum(x[u] * (1 - x[v]) + x[v] * (1 - x[u]) for u, v in list(graph.edge_list()))


import matplotlib.pyplot as plt
def plot_result(G, x, n):
    #colors = ["tab:grey" if i == 0 else "tab:purple" for i in x]
    colors = ["r" if x[i] == 0 else "c" for i in range(n)]

    pos, default_axes = rx.spring_layout(G), plt.axes(frameon=True)
    rx.visualization.mpl_draw(G, node_color=colors, node_size=100, alpha=0.8, pos=pos)


