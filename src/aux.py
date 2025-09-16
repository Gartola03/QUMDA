
import heapq

def UpdateProbabilityVectorCostFunctionWeighted(counts, observable, n, alpha=0.1, epsilon=1e-8):
    # Step 1: Expand (bitstring, cost) according to count
    samples_flat = []
    for bitstr, count in counts.items():
        state_int = int(bitstr, 2)
        cost = calculate_cost(state_int, observable)
        samples_flat.extend([(bitstr, cost)] * count)

    # Step 2: Sort by ascending cost (minimization)
    samples_flat.sort(key=lambda x: x[1])

    # Step 3: Select top α-fraction
    total_shots = len(samples_flat)
    num_to_keep = max(int(alpha * total_shots), 1)
    top_samples = samples_flat[:num_to_keep]

    # Step 4: Convert cost → fitness (lower cost = higher fitness)
    costs = [cost for _, cost in top_samples]
    max_cost = max(costs)
    fitnesses = [(max_cost - cost + epsilon) for cost in costs]  # higher fitness for lower cost
    total_fitness = sum(fitnesses)
    weights = [f / total_fitness for f in fitnesses]

    # Step 5: Weighted bit accumulation
    ST = [0.0] * n
    for (bitstr, _), weight in zip(top_samples, weights):
        for i in range(n):
            ST[i] += weight * int(bitstr[i])

    return ST


def UpdateProbabilityVectorCostFunctionAlpha(counts, observable, n, alpha=0.1):
    # Step 1: Expand (bitstring, cost) according to count
    samples_flat = []
    for bitstr, count in counts.items():
        state_int = int(bitstr, 2)
        cost = calculate_cost(state_int, observable)
        samples_flat.extend([(bitstr, cost)] * count)

    # Step 2: Sort by cost descending (maximize)
    samples_flat.sort(key=lambda x: x[1])

    # Step 3: Keep top α-fraction (at least 1 to avoid zero division)
    total_shots = len(samples_flat)
    num_to_keep = max(int(alpha * total_shots), 1)
    top_samples = samples_flat[:num_to_keep]

    # Step 4: Initialize bitwise accumulator
    ST = [0] * n

    # Step 5: Accumulate bit values
    for bitstr, _ in top_samples:
        for i in range(n):
            ST[i] += int(bitstr[i])

    # Step 6: Normalize to get probability vector
    ST = [x / num_to_keep for x in ST]

    return ST


def trunk(counts, observable, alpha):
    # Step 1: Expand into (state_int, cost) repeated by count
    samples_flat = []
    for state, count in counts.items():
        cost = calculate_cost(int(state, 2), observable)
        samples_flat.extend([(state, cost)] * count)

    # Step 2: Sort by cost descending (maximization)
    samples_flat.sort(key=lambda x: x[1])

    # Step 3: Keep top α fraction
    total_shots = sum(counts.values())
    num_to_keep = int(alpha * total_shots)
    top_samples = samples_flat[:num_to_keep]

    # Step 4: Compute average cost over selected samples
    expected_truncated = sum(cost for _, cost in top_samples) / num_to_keep

    return expected_truncated, top_samples


def update_distribution(self, qc, probability_vector):
    param_dict = {self.thetas[i]: 2*np.arcsin(np.sqrt(np.clip(probability_vector[i], 0.0, 1.0))) for i in range(self.n_qubits)}
    qc = qc.assign_parameters(param_dict)
    return qc

def UpdateProbabilityVector(counts, SelectedAmount):

    candidates = heapq.nlargest(SelectedAmount, counts, key=counts.get)

    # Get the number of qubits from the bitstring length (assuming all bitstrings are of the same length)
    n = len(candidates[0])  # Assuming all candidates are of the same length

    # Initialize ST as a zero vector of length n (number of qubits)
    ST = [0] * n

    # Concatenate all top candidates into a single string
    all_candidates_str = ''.join(candidates)

    # Accumulate the sums for each bit position across the concatenated candidates
    for i in range(len(all_candidates_str)):
        ST[i % n] += int(all_candidates_str[i])  # Convert binary character to integer

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


def energy_cost_function(samples, hamiltonian):
    total_counts = sum(samples.values())
    expected_energy = 0.0

    for bitstring, count in samples.items():
        # Convert bitstring to int or other needed format
        bit_val = int(bitstring, 2)
        # Calculate cost (energy) for this bitstring
        cost = calculate_cost(bit_val, hamiltonian)
        # Weight by probability of the bitstring
        expected_energy += (count / total_counts) * cost

    return expected_energy

def fitness_cost_function(samples, graph_sample):
    total_counts = sum(samples.values())
    expected_fitness = 0.0

    for bitstring, count in samples.items():
        # Convert bitstring to int or other needed format
        bit_val = to_bitstring(int(bitstring, 2), len(graph_sample))
        # Calculate cost (energy) for this bitstring
        bit_val.reverse()

        #Cut value
        cut_value= evaluate_sample(bit_val, graph_sample)
        # Weight by probability of the bitstring
        expected_fitness += (count / total_counts) * cut_value

    return expected_fitness


def random_search_minimizer(init_params, candidate, transpile_obs, estimator, cost_function,ITERATION_MAX):
    min_cost = float('inf')

    for i in range(ITERATION_MAX):
        cost = cost_function(init_params, candidate, transpile_obs, estimator)

        if min_cost > cost:
            result = init_params

        for i in range(len(init_params)):
            init_params[i] = np.random.uniform(-np.pi, np.pi)

    return result


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

def get_population_size(p):
    return 10*2*p



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


def save_figure(PLOT_DIR, name_str, fig):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, f"{name_str}.png"), bbox_inches="tight")



def save_result(graph_name, qaoa_type_name, objective_func_vals, result_x, counts, best_solution_string, cut_value, elapsed, final_distribution, save_dir):
    base_file_name = graph_name + ': ' + qaoa_type_name
    file_name = base_file_name
    counter = 1

    # Ensure unique file name if one already exists
    file_path = os.path.join(save_dir, file_name + ".json")
    while os.path.exists(file_path):
        file_name = f"{base_file_name}_{counter}"
        file_path = os.path.join(save_dir, file_name + ".json")
        counter += 1

    # Prepare data
    obj_func_cost = [float(i) for i in objective_func_vals]
    results_x = [float(i) for i in result_x]

    qaoa_results = {
        "file_name": file_name,
        "objective_func_cost": obj_func_cost,
        "sample_distribution": counts,
        "optimal_parameters": results_x,
        "beste_result_bitstring": best_solution_string,
        "value_of_cut": cut_value,
        "elapsed_time_seconds": round(elapsed, 4),
        "final_distribution": final_distribution
    }

    # Write to file
    with open(file_path, "w") as f:
        json.dump(qaoa_results, f, indent=4)

    print("Saved")

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


def to_bitstring_r(integer, num_bits, index):
  result = list(np.binary_repr(integer, width=num_bits))
  bit_list = [int(digit) for digit in result]
  insert_pos = len(bit_list) - index
  bit_list.insert(insert_pos, 0)
  return bit_list

from typing import Sequence
def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(list(graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
    return sum(x[u] * (1 - x[v]) + x[v] * (1 - x[u]) for u, v in list(graph.edge_list()))


import matplotlib.pyplot as plt
def plot_result(G, x, n):
    #colors = ["tab:grey" if i == 0 else "tab:purple" for i in x]
    colors = ["r" if x[i] == 0 else "g" for i in range(n)]

    pos, default_axes = rx.spring_layout(G), plt.axes(frameon=True)
    rx.visualization.mpl_draw(G, node_color=colors, node_size=200, with_labels=True, alpha=0.8, pos=pos)

def plot_graph_assign(G, highlight_node=None):
    # Default color for all nodes
    colors = []

    for node in G.nodes():
        if node == highlight_node:
            colors.append("tab:red")  # Highlight color
        else:
            colors.append("tab:blue")  # Default color

    pos = rx.spring_layout(G)
    default_axes = plt.axes(frameon=True)

    rx.visualization.mpl_draw(G, node_color=colors, node_size=200, with_labels=True, width=1, pos=pos)




#GRAPH QUBO


def get_qubo_matrix_rustworkx(graph):
    """Return QUBO matrix for Max-Cut problem from a rustworkx.PyGraph."""
    n = graph.num_nodes()
    Q = np.zeros((n, n))

    for edge in graph.edge_list():
        i, j = edge
        w = graph.get_edge_data(i, j)
        weight = w if w is not None else 1.0

        Q[i, i] += weight
        Q[j, j] += weight
        Q[i, j] -= 2 * weight  # Only upper/lower triangle filled asymmetrically here

    return Q

def symmetrize_matrix(Q):
    """Return symmetric version of Q by averaging with its transpose."""
    return (Q + Q.T) / 2

def qubo_to_rustworkx_graph_s(Q):
    n = Q.shape[0]
    graph = rx.PyGraph()

    for i in range(n):
        graph.add_node(i)

    for i in range(n):
        for j in range(i+1, n):
            weight = -Q[i, j]  # take upper triangle only
            if weight != 0:
                graph.add_edge(i, j, weight)

    return graph

def qubo_to_rustworkx_graph_a(Q):
    n = Q.shape[0]
    graph = rx.PyGraph()

    for i in range(n):
        graph.add_node(i)

    Q_sym = (Q + Q.T) / 2  # symmetrize

    for i in range(n):
        for j in range(i+1, n):
            weight = -Q_sym[i, j]
            if weight != 0:
                graph.add_edge(i, j, weight)

    return graph


def reduce_qubo(Q, fixed_vars):
    """
    Reduces a QUBO matrix by assigning selected variables to 0.

    Args:
        Q (np.ndarray): Original QUBO matrix (upper triangular, n x n).
        fixed_vars (list of int): Indices of variables to be fixed to 0.

    Returns:
        Q_reduced (np.ndarray): Reduced QUBO matrix (on free variables).
        const_offset (float): Constant contribution from fixed vars.
        mapping (dict): Mapping from reduced indices to original indices.
    """
    Q = np.array(Q)
    n = Q.shape[0]
    all_vars = set(range(n))
    free_vars = sorted(list(all_vars - set(fixed_vars)))

    # Build reduced QUBO
    Q_reduced = Q[np.ix_(free_vars, free_vars)]

    # Compute constant contribution from fixed vars
    const_offset = 0.0
    for i in fixed_vars:
        const_offset += Q[i, i]  # Linear term
        for j in free_vars:
            const_offset += Q[i, j] * 0  # x_i * x_j → 0 if x_i = 0
        for j in fixed_vars:
            if i < j:
                const_offset += Q[i, j] * 0  # x_i * x_j → 0

    # Mapping from reduced indices to original indices
    index_map = {new_i: orig_i for new_i, orig_i in enumerate(free_vars)}

    return Q_reduced, const_offset, index_map

def reduce_qubo_fixed_ones(Q, fixed_vars):

  Q = np.array(Q)
  n = Q.shape[0]
  all_vars = set(range(n))
  free_vars = sorted(list(all_vars - set(fixed_vars)))

  # Build reduced QUBO (submatrix of free variables)
  Q_reduced = Q[np.ix_(free_vars, free_vars)].copy()

  # Initialize constant offset and linear offset
  const_offset = 0.0
  linear_offset = np.zeros(len(free_vars))

  # Process fixed variables
  for i in fixed_vars:
      # Add the linear contribution
      const_offset += Q[i, i]

      # Add interaction with free variables to linear offset
      for idx_j, j in enumerate(free_vars):
          linear_offset[idx_j] += Q[i, j]

      # Add interaction with other fixed vars to constant offset
      for j in fixed_vars:
          if i < j:
              const_offset += Q[i, j]

  # Incorporate linear offset into diagonal of reduced QUBO
  for idx_j in range(len(free_vars)):
      Q_reduced[idx_j, idx_j] += linear_offset[idx_j]

  # Mapping from reduced indices to original indices
  index_map = {new_i: orig_i for new_i, orig_i in enumerate(free_vars)}

  return Q_reduced, const_offset, index_map


def qubo_to_ising(Q_qubo):
    """
    Convert QUBO matrix to Ising Hamiltonian form:
      H(z) = sum_ij J_ij z_i z_j + sum_i b_i z_i + const

    Args:
        Q_qubo: numpy array, symmetric QUBO matrix

    Returns:
        J: numpy array, Ising couplings matrix (zero diagonal)
        b: numpy array, Ising bias vector
        const: float, constant offset energy term (can be ignored in optimization)
    """
    # Ensure symmetry of Q
    Q_sym = (Q_qubo + Q_qubo.T) / 2

    # Compute Ising couplings J_ij = Q_ij / 4 for i != j
    J = Q_sym / 4
    np.fill_diagonal(J, 0)

    # Compute biases b_i = -1/2 * sum_j Q_ij
    b = -0.5 * Q_sym.sum(axis=1)

    # Constant term = 1/4 * sum_ij Q_ij
    const = 0.25 * Q_sym.sum()

    return J, b, const

def symmetric_to_upper_double(J):
    """
    Given symmetric matrix J (with zero diagonal),
    return an upper-triangular matrix with entries 2 * J_ij,
    zeros elsewhere.

    Args:
        J: numpy array, symmetric matrix with zero diagonal

    Returns:
        J_upper_double: numpy array, upper-triangular matrix with doubled off-diagonal entries
    """
    # Double J and keep only upper triangle (excluding diagonal)
    J_upper_double = np.triu(2 * J, k=1)
    return J_upper_double
