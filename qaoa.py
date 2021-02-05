from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.optimization.applications.ising import max_cut
from qiskit.compiler import transpile, assemble
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter

class QAOA():
    def __init__(self, betas, gammas, number_of_qubits, layers, w, graph):
        self.qreg = QuantumRegister(number_of_qubits, name = 'q')
        self.creg = ClassicalRegister(number_of_qubits, name = 'c')
        self.qaoa = QuantumCircuit(self.qreg, self.creg)

        self.layers = layers
        self.number_of_qubits = number_of_qubits
        self.w = w
        self.graph = graph

        self.qaoa.h(range(self.number_of_qubits))
        self.qaoa.barrier

        self.betas = betas
        self.gammas = gammas

        for layer in range(self.layers):
            for i in range(number_of_qubits):
                for j in range(number_of_qubits):
                    if self.w[i,j] != 0:
                        self.qaoa += self.ZZ(i, j, self.gammas[layer], self.qreg, self.creg)
            
            self.qaoa.barrier()

            for qubit in np.arange(self.number_of_qubits):
                self.qaoa.rx(self.betas[layer], qubit)
        
        self.qaoa.barrier()

        self.qaoa.measure(range(self.number_of_qubits), self.creg)
        self.counts = execute(self.qaoa, Aer.get_backend('qasm_simulator'), shots=1024).result().get_counts()
        

    def ZZ(self, qubit1, qubit2, gamma, qreg, creg):
        self.q_encode = QuantumCircuit(qreg, creg)
        self.q_encode.cx(qubit1, qubit2)
        self.q_encode.u1(gamma, qubit2)
        self.q_encode.cx(qubit1, qubit2)
        return self.q_encode

    def get_expected_value(self):
        avr_c = 0
        for sample in list(self.counts.keys()):
            y = [int(num) for num in list(sample)]
            tmp_eng = self.cost_hamiltonian(y) + self.get_offset()
            avr_c += self.counts[sample] * tmp_eng
        energy_expectation = avr_c/len(self.counts)
        return energy_expectation

    def cost_hamiltonian(self, x):
        spins = []
        for i in x[::-1]:
            spins.append(int(i))
        total_energy = 0
        for i in range(self.number_of_qubits):
            for j in range(self.number_of_qubits):
                if self.w[i,j] != 0:
                    total_energy += self.sigma(spins[i]) * self.sigma(spins[j])
        total_energy /= 4
        return total_energy

    def sigma(self, z):
        if z == 0:
            value = 1
        elif z == 1:
            value = -1
        return value

    def get_offset(self):
        return -self.graph.number_of_edges()/2

    def best_cost_brute(self):
        best_cost = 0
        for b in range(2**self.number_of_qubits):
            x = [int(t) for t in reversed(list(bin(b)[2:].zfill(self.number_of_qubits)))]
            cost = 0
            for i in range(self.number_of_qubits):
                for j in range(self.number_of_qubits):
                    cost += self.w[i,j] * x[i] * (1-x[j])
            if best_cost < cost:
                best_cost = cost
        return best_cost
