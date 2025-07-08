import itertools
import numpy as np
from tqdm import tqdm
from dwdynamics import ComplexDynamicsProblem
import networkx as nx
import dimod

# Valeurs possibles
valeurs_b = [-1, 0, 1]
valeurs_c = [-1, 0,1+1j,-1+1j,1j,-1j, -1-1j]


# Génération de combinaisons
nombre_de_variables_b = 2
nombre_de_variables_c = 1

combinaisons_b = itertools.product(valeurs_b, repeat=nombre_de_variables_b)

my_matrices = []

for b in tqdm(combinaisons_b, desc="b"):
    combinaisons_c = itertools.product(valeurs_c, repeat=nombre_de_variables_c)

    for c in tqdm(combinaisons_c, desc="c", leave=False):
        M = np.array([
            [b[0], c[0]],
            [np.conj(c[0]), b[1]]
        ], dtype=complex)

        if True: #np.allclose(M @ M, np.eye(2), atol=1e-6):
            problem = ComplexDynamicsProblem(
                hamiltonian=0.5 * np.pi * M,
                initial_state=np.array([1, 0]),
                times=(0, 1),
                num_bits_per_var=2
            )
            qubo = problem.qubo()
            if nx.is_connected(dimod.to_networkx_graph(qubo)):
                print("found !!")
                print(M)
                my_matrices.append(M)

print(f"{len(my_matrices)} matrices trouvées")