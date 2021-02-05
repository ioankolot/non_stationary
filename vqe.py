from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.optimization.applications.ising import max_cut
from qiskit.compiler import transpile, assemble
from qiskit.visualization import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
from qiskit.circuit.library import EfficientSU2


class VQE():
    def __init__(self, number_of_qubits, w, graph, thetas):

        self.number_of_qubits = number_of_qubits
        self.w = w
        self.graph = graph
        self.init_thetas = thetas[0]
        self.first_layer_thetas = thetas[1]


        self.qreg = QuantumRegister(self.number_of_qubits, name = 'q')
        self.creg = ClassicalRegister(self.number_of_qubits, name = 'c')
        self.vqe = QuantumCircuit(self.qreg, self.creg)

        for qubit in range(self.number_of_qubits):
            self.vqe.ry(self.init_thetas[qubit], qubit)

        self.vqe.barrier()

        for qubit1 in range(self.number_of_qubits):
            for qubit2 in range(self.number_of_qubits):
                if qubit1 > qubit2:
                    self.vqe.cz(qubit1, qubit2)

        for qubit in range(self.number_of_qubits):
            self.vqe.ry(self.first_layer_thetas[qubit], qubit)

        self.vqe.barrier()

        self.vqe.measure(range(self.number_of_qubits), self.creg)
        self.counts = execute(self.vqe, Aer.get_backend('qasm_simulator'), shots = 10000).result().get_counts()

    def get_expected_value(self):
        avr_c = 0
        for sample in list(self.counts.keys()):
            y = [int(num) for num in list(sample)]
            tmp_eng = self.cost_hamiltonian(y) + self.get_offset()
            avr_c += self.counts[sample] * tmp_eng
        energy_expectation = avr_c/10000
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

    def probability_of_optimal(self):
        optimal_solution = self.best_cost_brute()
        energies = self.exact_counts()
        total_counts_of_optimal = 0
        for energy in energies:
            if energy == -optimal_solution:
                total_counts_of_optimal += 1
        return total_counts_of_optimal/10000


    def exact_counts(self):
        energies = []
        for sample in list(self.counts.keys()):
            y = [int(num) for num in list(sample)]
            tmp_eng = self.cost_hamiltonian(y) + self.get_offset()
            for num in range(self.counts[sample]):
                energies.append(tmp_eng)
        energies.sort(reverse=False)
        return energies


