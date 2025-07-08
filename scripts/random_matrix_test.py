import itertools
import numpy as np
from tqdm import tqdm
from dwdynamics import ComplexDynamicsProblem
import networkx as nx
import dimod

valeurs_b = [-1, 0, 1,0.5,-0.5]
valeurs_c = [0.5+0.5j,-0.5j,-0.5j+0.5j,-1, 0,1+1j,-1+1j,1j,0.5+1j,-1+0.5j,1+0.5j]

nombre_de_variables_b = 4
nombre_de_variables_c = 6

combinaisons_b = itertools.product(valeurs_b, repeat=nombre_de_variables_b)
combinaisons_c = itertools.product(valeurs_c, repeat=nombre_de_variables_c)

my_matrices = []

for b in tqdm(combinaisons_b, total=len(valeurs_b)**nombre_de_variables_b):
    combinaisons_c = itertools.product(valeurs_c, repeat=nombre_de_variables_c)
    for c in tqdm(combinaisons_c, total=len(valeurs_c)**nombre_de_variables_c,leave=False):
        M = np.array([
            [b[0], c[0], c[1], c[2]],
            [np.conj(c[0]), b[1], c[3], c[4]],
            [np.conj(c[1]), np.conj(c[3]), b[2], c[5]],
            [np.conj(c[2]), np.conj(c[4]), np.conj(c[5]), b[3]]])
        if np.allclose(M @ M, np.eye(4), atol=1e-6):
            problem = ComplexDynamicsProblem(
                hamiltonian=0.5 * np.pi * M,      # Evolution under hamiltonian 0.5 * np.pi * sigma_y
                initial_state = np.array([1,0,0,0]),              # Starting at initial state |0>,
                times=tuple(range(2)),             # With six time points 0,1,2
                num_bits_per_var=3               # And two bits of precision per variable
            )
            qubo =problem.qubo()
            if nx.is_connected(dimod.to_networkx_graph(qubo)):
                print("found !!")
                print(M)

                my_matrices.append(M)

print(my_matrices)