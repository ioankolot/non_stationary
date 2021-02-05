import numpy as np
import networkx as nx 
from qaoa import QAOA
import scipy.optimize
import random

graph = nx.random_regular_graph(4, 12, seed=10)
number_of_qubits = 12
w = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))


nelder_old_betas = False
nelder_old_gammas = False
def nelder_mead(x):
    betas=[x[0]]
    gammas=[x[1]]
    if nelder_old_betas:
        betas = nelder_old_betas + betas
        gammas = nelder_old_gammas + gammas
    graph_qaoa = QAOA(betas, gammas, number_of_qubits, 3, w, graph)
    energy = graph_qaoa.get_expected_value()
    return float(energy)


graph_dataset1_layer4 = [[] for _ in range(50)]
cntr = 0

for num in range(1):
    minimum_energy = 0
    nelder_old_betas = [2.4170641221158116]
    nelder_old_gammas = [4.9654617843417075]
    for beta in np.linspace(0, np.pi, 6):
        for gamma in np.linspace(0, 2*np.pi, 10):
            number_of_qubits = len(graph.nodes())
            w = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))
            minimum_energy_object = scipy.optimize.minimize(nelder_mead, x0=(beta, gamma), method='Nelder-Mead')
            if minimum_energy_object.fun < minimum_energy:
                minimum_energy = minimum_energy_object.fun
                optimal_beta = minimum_energy_object.x[0]
                optimal_gamma = minimum_energy_object.x[1]
                print('{}'.format(cntr))
    print("For the {} graph with Nelder Mead the minimum energy is {} with optimal beta:{} and optimal gamma: {}".format(cntr, minimum_energy, optimal_beta, optimal_gamma))           
    graph_dataset1_layer4[cntr].append(minimum_energy)
    graph_dataset1_layer4[cntr].append(optimal_beta)
    graph_dataset1_layer4[cntr].append(optimal_gamma)