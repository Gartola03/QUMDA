import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

class QAOA:

  def __init__(self, p: int, n_qubits: int, cost_hamiltonian):

    #Parameters
    self.p = p
    self.n_qubits = n_qubits
    self.gammas = self.p * [Parameter(f'γ_{i}') for i in range(p)]
    self.betas =  self.p * [Parameter(f'β_{i}') for i in range(p)]
    #ParameterVector("β", self.reps * num_mixer)

    #Cost hamiltonian
    self.cost_hamiltonian = cost_hamiltonian

    #Circuit
    self.qc = self.construct_circuit()



  def construct_circuit(self):

    qc = QuantumCircuit(self.n_qubits, self.n_qubits)

    # Initial state: uniform superposition
    qc.h(range(self.n_qubits))

    for i in range(self.p):
      # Apply cost Hamiltonian evolution
      qc = self._cost_hamiltonian(qc, self.gammas[i])

      # Apply mixing Hamiltonian (X rotation)
      qc = self._mixing_hamiltonian(qc, self.betas[i])

    qc.measure_all()
    return qc

  def _cost_hamiltonian(self, qc, gamma):
    for term, coeff in zip(self.cost_hamiltonian.paulis, self.cost_hamiltonian.coeffs):
      pauli_str = term.to_label()
      qubits = [i for i, p in enumerate(pauli_str) if p != 'I']

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

    return qc


  def _mixing_hamiltonian(self, qc, beta):
    qc.rx(2 * beta, range(self.n_qubits))
    return qc
