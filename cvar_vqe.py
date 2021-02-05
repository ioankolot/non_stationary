import math
from vqe import VQE
import networkx as nx
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

decay_factor = 0.12
alpha = 1
alpha_minimum = 0.1

graph = nx.random_regular_graph(4, 16, seed=10)
#graph = nx.dense_gnm_random_graph(8, 40, seed=10)
#graph = nx.gnp_random_graph(8, p=0.3, seed=10, directed=False) #random Erdos-Renyi Graph
number_of_qubits = 16
w = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))


def CVaR(percentage, energies):
    len_list = len(energies)
    ak = math.ceil(len_list * percentage)
    cvar = 0
    for sample in range(ak):
        cvar += energies[sample]
    return cvar/ak



#We have to test how the decay CVaR behaves in respect to other constant CVaRa with different values of a=const
def constant_cvar_function(x, alfa):
    constant_thetas_init = [x[i] for i in range(number_of_qubits)]
    constant_thetas_first_layer = [x[i+number_of_qubits] for i in range(number_of_qubits)]
    constant_thetas = [constant_thetas_init, constant_thetas_first_layer]
    vqe_const = VQE(number_of_qubits, w, graph, constant_thetas)
    vqe_const_counts = vqe_const.counts
    const_energies = []
    for sample in list(vqe_const_counts.keys()):
        y = [int(num) for num in list(sample)]
        tmp_eng = vqe_const.cost_hamiltonian(y) + vqe_const.get_offset()
        for num in range(vqe_const_counts[sample]):
            const_energies.append(tmp_eng)
    const_energies.sort(reverse=False)
    const_cvar = CVaR(alfa, const_energies)
    return const_cvar



def decay_cvar_function(x, alpha):
    decaying_thetas_init= [x[i] for i in range(number_of_qubits)]
    decaying_thetas_first_layer = [x[i+number_of_qubits] for i in range(number_of_qubits)]
    decaying_thetas = [decaying_thetas_init, decaying_thetas_first_layer]
    decaying_vqe = VQE(number_of_qubits, w, graph, decaying_thetas)
    decaying_vqe_counts = decaying_vqe.counts
    decaying_energies = []
    for sample in list(decaying_vqe_counts.keys()):
        y = [int(num) for num in list(sample)]
        tmp_eng = decaying_vqe.cost_hamiltonian(y) + decaying_vqe.get_offset()
        for num in range(decaying_vqe_counts[sample]):
            decaying_energies.append(tmp_eng)
    decaying_energies.sort(reverse=False)

    decaying_cvar = CVaR(alpha, decaying_energies)
    return decaying_cvar


'''
thetas_init = [np.random.uniform(0, 2*np.pi) for _ in range(2*number_of_qubits)]
minimum_energy_object = scipy.optimize.minimize(constant_cvar_function, x0=tuple(thetas_init), args=(alfa=0.1), method='COBYLA')
objective_value = minimum_energy_object.fun
print(objective_value)
optimal_thetas = [0 for _ in range(2*number_of_qubits)]
for num in range(2*number_of_qubits):
    optimal_thetas[num] = minimum_energy_object.x[num]

thetas = [optimal_thetas[0:number_of_qubits], optimal_thetas[number_of_qubits:]]
vqe = VQE(number_of_qubits, w, graph, thetas)
print(vqe.best_cost_brute())
print(vqe.exact_counts())
print(np.mean(vqe.exact_counts()[0:2000]))
print(vqe.probability_of_optimal())
print(len(vqe.exact_counts()))
print(vqe.get_expected_value())
'''





def decay_optimization(decay_parameter = decay_factor):
    #We begin by initializing the same random parameters for all optimizers.
    thetas_init = [np.random.uniform(0, 2*np.pi) for _ in range(2*number_of_qubits)]
    thetas_init_01 = thetas_init
    thetas_init_02 = thetas_init
    thetas_init_05 = thetas_init
    thetas_init_expectation_value = thetas_init
    thetas_init_decaying = thetas_init

    #We set the global parameter alpha that will decay slowly
    global alpha

    const_cvar_01 = []   #We save only the objective values for plotting
    const_cvar_02 = []
    const_cvar_05 = []
    expectation_values = []
    decaying_cvar = []


    constant_optimization_01 = [] #In these lists we save the objective function values along with the optimal parameters
    constant_optimization_02 = []
    constant_optimization_05 = []
    constant_optimization_expectation_value = []
    decaying_optimization = []


    constant_probabilities_01 = []  # In these lists we save the probabilities of sampling the objective function.
    constant_probabilities_02 = []
    constant_probabilities_05 = []
    constant_probabilities_expectation_value = []
    decaying_probabilities = []


    '''
    minimum_energy_object = scipy.optimize.minimize(decay_cvar_function, x0=tuple(thetas_init), args=(alpha), method='COBYLA')
    objective_value = minimum_energy_object.fun
    print(f'The optimization of the expectation value is {objective_value} with alpha = {alpha}' )

    #We save the optimal parameters for the decaying optimization that will be used as initial points for the next optimization.
    for num in range(2*number_of_qubits):
        optimal_thetas[num] = minimum_energy_object.x[num]
    decaying_optimization.append([objective_value, optimal_thetas])
    thetas_init_decaying = optimal_thetas
    '''
    constant_cvar_optimal_thetas_01 = [0 for _ in range(2*number_of_qubits)]
    constant_cvar_optimal_thetas_02 = [0 for _ in range(2*number_of_qubits)]
    constant_cvar_optimal_thetas_05 = [0 for _ in range(2*number_of_qubits)]
    expectation_value_optimal_thetas = [0 for _ in range(2*number_of_qubits)]
    decaying_optimal_thetas = [0 for _ in range(2*number_of_qubits)]

    k = 0
    while k < 20: #We slowly decay alpha from alpha = 1 to alpha = 0.1
        alpha -= alpha*decay_parameter
        if alpha < alpha_minimum:
            alpha = alpha_minimum

        #We minize all the different obejctive functions
        decaying_optimization_object = scipy.optimize.minimize(decay_cvar_function, x0 = tuple(thetas_init_decaying), args=(alpha), method='COBYLA', options={'maxiter':100})
        decaying_objective_value = decaying_optimization_object.fun
        decaying_cvar.append(decaying_objective_value)
        print(f' On the decaying optimization on {k} iteration with alpha = {alpha} the objective value is {decaying_objective_value}')
        for num in range(2*number_of_qubits):
            decaying_optimal_thetas[num] = decaying_optimization_object.x[num]

        thetas_init_decaying = decaying_optimal_thetas

        thetas_decaying = [decaying_optimal_thetas[0:number_of_qubits], decaying_optimal_thetas[number_of_qubits:]]
        decaying_vqe = VQE(number_of_qubits, w, graph, thetas_decaying)

        decaying_probability = decaying_vqe.probability_of_optimal()
        decaying_probabilities.append(decaying_probability)
        print(f'and the probability of obtaining the optimal solution is {decaying_probability}')        

        constant_cvar_01_object = scipy.optimize.minimize(constant_cvar_function, x0 = tuple(thetas_init_01), args=(0.1), method='COBYLA',  options={'maxiter':100})
        constant_01_objective_value = constant_cvar_01_object.fun
        const_cvar_01.append(constant_01_objective_value)
        print(f'On the {k} iteration with constant alpha = 0.1 the objective values is {constant_01_objective_value}')
        for num in range(2*number_of_qubits):
            constant_cvar_optimal_thetas_01[num] = constant_cvar_01_object.x[num]

        thetas_init_01 = constant_cvar_optimal_thetas_01

        constant_cvar_thetas_01 = [constant_cvar_optimal_thetas_01[0:number_of_qubits], constant_cvar_optimal_thetas_01[number_of_qubits:]]
        constant_01_vqe = VQE(number_of_qubits, w, graph, constant_cvar_thetas_01)

        constant_cvar_01_probability = constant_01_vqe.probability_of_optimal()
        constant_probabilities_01.append(constant_cvar_01_probability)
        print(f'and the probability of obtaining the optimal solution is {constant_cvar_01_probability}')

        constant_cvar_02_object = scipy.optimize.minimize(constant_cvar_function, x0 = tuple(thetas_init_02), args=(0.2), method='COBYLA',  options={'maxiter':100})
        constant_02_objective_value = constant_cvar_02_object.fun
        const_cvar_02.append(constant_02_objective_value)
        print(f'On the {k} iteration with constant alpha = 0.2 the objective values is {constant_02_objective_value}')
        for num in range(2*number_of_qubits):
            constant_cvar_optimal_thetas_02[num] = constant_cvar_02_object.x[num]

        thetas_init_02 = constant_cvar_optimal_thetas_02

        constant_cvar_thetas_02 = [constant_cvar_optimal_thetas_02[0:number_of_qubits], constant_cvar_optimal_thetas_02[number_of_qubits:]]
        constant_02_vqe = VQE(number_of_qubits, w, graph, constant_cvar_thetas_02)

        constant_cvar_02_probability = constant_02_vqe.probability_of_optimal()
        constant_probabilities_02.append(constant_cvar_02_probability)
        print(f'and the probability of obtaining the optimal solution is {constant_cvar_02_probability}')


        constant_cvar_05_object = scipy.optimize.minimize(constant_cvar_function, x0 = tuple(thetas_init_05), args=(0.5), method='COBYLA',  options={'maxiter':100})
        constant_05_objective_value = constant_cvar_05_object.fun
        const_cvar_05.append(constant_05_objective_value)
        print(f'On the {k} iteration with constant alpha = 0.5 the objective values is {constant_05_objective_value}')
        for num in range(2*number_of_qubits):
            constant_cvar_optimal_thetas_05[num] = constant_cvar_05_object.x[num]

        thetas_init_05 = constant_cvar_optimal_thetas_05
    
        constant_cvar_thetas_05 = [constant_cvar_optimal_thetas_05[0:number_of_qubits], constant_cvar_optimal_thetas_05[number_of_qubits:]]
        constant_05_vqe = VQE(number_of_qubits, w, graph, constant_cvar_thetas_05)

        constant_cvar_05_probability = constant_05_vqe.probability_of_optimal()
        constant_probabilities_05.append(constant_cvar_05_probability)
        print(f'and the probability of obtaining the optimal solution is {constant_cvar_05_probability}')

        expectation_value_object = scipy.optimize.minimize(constant_cvar_function, x0 = tuple(thetas_init_expectation_value), args=(1), method='COBYLA',  options={'maxiter':100})
        expectation_value = expectation_value_object.fun
        expectation_values.append(expectation_value)
        print(f'On the {k} iteration the expectation value is {expectation_value}')
        for num in range(2*number_of_qubits):
            expectation_value_optimal_thetas[num] = expectation_value_object.x[num]

        thetas_init_expectation_value = expectation_value_optimal_thetas

        expectation_value_thetas = [expectation_value_optimal_thetas[0:number_of_qubits], expectation_value_optimal_thetas[number_of_qubits:]]
        expectation_value_vqe = VQE(number_of_qubits, w, graph, expectation_value_thetas)

        expectation_value_probability = expectation_value_vqe.probability_of_optimal()
        constant_probabilities_expectation_value.append(expectation_value_probability)
        print(f'and the probability of obtaining the optimal solution is {expectation_value_probability}')

        k += 1

    return constant_probabilities_expectation_value, constant_probabilities_05, constant_probabilities_02, constant_probabilities_01, decaying_probabilities


expectation_probabilities, constant_05_probabilities, constant_02_probabilities, constant_01_probabilities, decaying_probabilities = decay_optimization()

print(alpha)

decaying_alphas = [1]
alpha = 1
k=0
while k < 20:
    alpha -= alpha*decay_parameter
    if alpha < alpha_minimum:
        alpha = alpha_minimum
    decaying_alphas.append(alpha)
    k += 1

plt.plot(expectation_probabilities,  color='black')
plt.plot(constant_05_probabilities, color='green')
plt.plot(constant_02_probabilities, color='red')
plt.plot(constant_01_probabilities, color='yellow')
plt.plot(decaying_probabilities, color='blue')
plt.show()