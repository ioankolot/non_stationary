import math
from qaoa import QAOA
import networkx as nx
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#We are gonna define a new objective function that smoothly decays from being the expectation value to becoming the CVaR with a=0.1

#First of all we define a decay factor 

decay_factor = 0.12
alpha = 1
alpha_minimum = 0.1


graph = nx.random_regular_graph(3, 10, seed=10)
number_of_qubits = 10
w = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))

#The CVaR objective function is defined as :

def CVaR(alpha, energies):
    len_list = len(energies)
    ak = math.ceil(len_list * alpha)
    cvar = 0
    for sample in range(ak):
        cvar += energies[sample] / ak
    return cvar

def cvar_fun(x):
    betas_const = [x[0], x[1], x[2], x[3], x[4]]
    gammas_const = [x[5], x[6], x[7], x[8], x[9]]
    qaoa= QAOA(betas_const, gammas_const, number_of_qubits, 5, w, graph)
    qaoa_counts = qaoa.counts
    energies = []
    for sample in list(qaoa_counts.keys()):
        y = [int(num) for num in list(sample)]
        tmp_eng = qaoa.cost_hamiltonian(y) + qaoa.get_offset()
        energies.append(tmp_eng)
    energies.sort(reverse=False)
    cvar = CVaR(0.1, energies)
    return cvar

def decay_cvar(x, alpha = alpha):
    betas = [x[0], x[1], x[2], x[3], x[4]]
    gammas = [x[5], x[6], x[7], x[8], x[9]]
    qaoa= QAOA(betas, gammas, number_of_qubits, 5, w, graph)
    qaoa_counts = qaoa.counts
    energies = []
    for sample in list(qaoa_counts.keys()):
        y = [int(num) for num in list(sample)]
        tmp_eng = qaoa.cost_hamiltonian(y) + qaoa.get_offset()
        energies.append(tmp_eng)
    energies.sort(reverse=False)
    cvar = CVaR(alpha, energies)

    return cvar

#print(decay_cvar([0.7958146707098828, 2.8567258960761395]))


#We are gonna optimize the Decay-CVaR objective function


def decay_optimization(decay_parameter = decay_factor, alpha = alpha):
    betas = [np.random.uniform(0, np.pi/2) for _ in range(5)]
    gammas = [np.random.uniform(0, np.pi) for _ in range(5)]

    print(alpha)
    print(betas, gammas)
    const_betas = betas
    const_gammas = gammas
    const_cvar = []
    optimal_betas = [[] for _ in range(5)]
    optimal_gammas = [[] for _ in range(5)]

    decaying_optimization = []
    minimum_energy_object = scipy.optimize.minimize(decay_cvar, x0 = (betas[0], betas[1], betas[2], betas[3], betas[4], gammas[0], gammas[1], gammas[2], gammas[3], gammas[4]),args = (alpha), method='COBYLA')
    objective_value = minimum_energy_object.fun
    print(objective_value)
    optimal_betas[0] = minimum_energy_object.x[0]
    optimal_betas[1] = minimum_energy_object.x[1]
    optimal_betas[2] = minimum_energy_object.x[2]
    optimal_betas[3] = minimum_energy_object.x[3]
    optimal_betas[4] = minimum_energy_object.x[4]

    optimal_gammas[0] = minimum_energy_object.x[5]
    optimal_gammas[1] = minimum_energy_object.x[6]
    optimal_gammas[2] = minimum_energy_object.x[7]
    optimal_gammas[3] = minimum_energy_object.x[8]
    optimal_gammas[4] = minimum_energy_object.x[9]         


    decaying_optimization.append([objective_value, optimal_betas, optimal_gammas])
    betas = optimal_betas
    gammas = optimal_gammas


    k = 0
    while k < 20:
        alpha -= alpha*decay_parameter
        print(alpha)
        if alpha < alpha_minimum:
            alpha = alpha_minimum
        minimum_energy_object = scipy.optimize.minimize(decay_cvar, x0 = (betas[0], betas[1], betas[2], betas[3], betas[4], gammas[0], gammas[1], gammas[2], gammas[3], gammas[4]), args = (alpha), method='COBYLA')
        objective_value = minimum_energy_object.fun
        print(objective_value)
        optimal_betas[0] = minimum_energy_object.x[0]
        optimal_betas[1] = minimum_energy_object.x[1]
        optimal_betas[2] = minimum_energy_object.x[2]
        optimal_gammas[0] = minimum_energy_object.x[3]
        optimal_gammas[1] = minimum_energy_object.x[4]
        optimal_gammas[2] = minimum_energy_object.x[5]  

        decaying_optimization.append([objective_value, optimal_betas, optimal_gammas])
        betas = optimal_betas
        gammas = optimal_gammas

        minimum_energy_object2 = scipy.optimize.minimize(cvar_fun, x0 = (const_betas[0], const_betas[1], const_betas[2], const_betas[3], const_betas[4], const_gammas[0], const_gammas[1], const_gammas[2], const_gammas[3], const_gammas[4]), method='COBYLA')
        objective_value2 = minimum_energy_object2.fun
        print(objective_value2)
        const_betas[0] = minimum_energy_object2.x[0]
        const_betas[1] = minimum_energy_object2.x[1]
        const_betas[2] = minimum_energy_object2.x[2]
        const_betas[3] = minimum_energy_object2.x[3]
        const_betas[4] = minimum_energy_object2.x[4]

        const_gammas[0] = minimum_energy_object2.x[5]
        const_gammas[1] = minimum_energy_object2.x[6]
        const_gammas[2] = minimum_energy_object2.x[7]
        const_gammas[3] = minimum_energy_object2.x[8]
        const_gammas[4] = minimum_energy_object2.x[9]

        const_cvar.append([objective_value2, const_betas, const_gammas])

        k += 1


    return decaying_optimization, const_cvar



kolo, kolo2 = decay_optimization()

print(kolo,'\n', kolo2)

# We'll work for alpha = [1, 0.5, 0.2, 0.15]

def create_landscape(alpha):
    energies = np.zeros((100,100))
    cvar = np.zeros((100,100))
    for i,beta in enumerate(np.linspace(0, np.pi, 100)):
        for j, gamma in enumerate(np.linspace(0, 2*np.pi, 100)):
#            qaoa = QAOA([beta], [gamma], number_of_qubits, 1, w, graph)
#            energies[i,j] = qaoa.get_expected_value()
            cvar[i,j] = decay_cvar([beta], [gamma], alpha)

    return cvar


'''
energies = create_landscape(0.1)
X, Y = np.meshgrid(np.linspace(0, np.pi, 100), (np.linspace(0, 2*np.pi, 100)))

fig = plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, energies)
plt.xlabel('beta')
plt.ylabel('gamma')
plt.show()


plt.contourf(X, Y, energies)
plt.colorbar()
plt.xlabel('beta')
plt.ylabel('gamma')
plt.show()

parameter_list = decay_optimization(0.2)

print(parameter_list)
'''


# A figure of merit for benchmarking our algorithm is the probability of sampling the optimal solution by the number of iterations

