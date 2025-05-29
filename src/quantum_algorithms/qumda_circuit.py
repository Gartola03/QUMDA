import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class QUMDA:

  def __init__(self, p: int, n_qubits: int, cost_hamiltonian):

    #Parameters
    if p > 0:
      self.p = p
    else:
      self.p = 1

    self.n_qubits = n_qubits
    self.gammas = [Parameter(f'γ_{i}') for i in range(self.p)]
    self.betas =  [Parameter(f'β_{i}') for i in range(self.p)]
    #self.thetas = self.n_qubits * [0.5]
    self.thetas = [Parameter(f'θ_{i}') for i in range(self.n_qubits)]
    #ParameterVector("β", self.reps * num_mixer)

    #Cost hamiltonian
    self.cost_hamiltonian = cost_hamiltonian

    #Parameterizable circuit
    self.qc = self.construct_circuit()


  def construct_circuit(self):

    qc = QuantumCircuit(self.n_qubits, self.n_qubits)

    # Initial state:
    qc = self.initial_state_layer(qc)

    for i in range(self.p):
      # Apply cost Hamiltonian evolution
      qc = self.cost_hamiltonian_layer(qc, self.gammas[i])

      # Apply mixing Hamiltonian (X rotation)
      qc = self.mixing_hamiltonian_layer(qc, self.betas[i])

    qc.measure_all()
    return qc

  def initial_state_layer(self, qc):
    for i in range(self.n_qubits):
        #theta = 2*np.arcsin(np.sqrt(self.thetas[i]))
        theta = self.thetas[i]
        qc.ry(theta, i)
    return qc

  def cost_hamiltonian_layer(self, qc, gamma):
    for term, coeff in zip(self.cost_hamiltonian.paulis, self.cost_hamiltonian.coeffs):
      pauli_str = term.to_label()

      qubits = [i for i, p in enumerate(pauli_str[::-1]) if p != 'I']

      if pauli_str.count('Z') == 2:
        # Two-qubit ZZ term
        i, j = qubits
        qc.cx(i, j)
        #If max-cut, not 2
        qc.rz(coeff.real * gamma, j)
        qc.cx(i, j)

      elif pauli_str.count('Z') == 1:
        # Single-qubit Z term (if present)
        i = qubits[0]
        #If max-cut, not 2
        qc.rz(coeff.real * gamma, i)

      qc.barrier()

    return qc


  def mixing_hamiltonian_layer(self, qc, beta):
    for i in range(self.n_qubits):
      #theta = 2 * np.arcsin(np.sqrt(self.thetas[i]))
      theta = self.thetas[i]
      qc.ry(-theta, i)
      qc.rz(-2* beta, i)
      qc.ry(theta, i)

    return qc

  def update_distribution(self, probability_vector):
    param_dict = {self.thetas[i]: 2 * np.arcsin(np.sqrt(np.clip(probability_vector[i], 0.0, 1.0))) for i in range(self.n_qubits)}
    qc = self.qc.assign_parameters(param_dict)
    return qc

  def update_distribution2(self, qc, probability_vector):
    param_dict = {self.thetas[i]: 2*np.arcsin(np.sqrt(np.clip(probability_vector[i], 0.0, 1.0))) for i in range(self.n_qubits)}
    qc = qc.assign_parameters(param_dict)
    return qc