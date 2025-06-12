import numpy as np
import os
import pickle
import qutip as qp
import math
import networkx as nx
import matplotlib.pyplot as plt
import functools as ft
import re


def random_matrix(dims, hermitian=True):
    A = np.random.uniform(-1, 1, dims) + 1.j * np.random.uniform(-1, 1, dims)
    if hermitian:
        return A + A.conj().T
    return A


def get_last_index(files: list[str])-> int:
    """
        Returns last index of the files specified in input files  
    """
    if files == []:
        return 0
    return max([int(re.findall('\d+(?=\.json)',file)[0]) for file in files])


"""
    Retrives an instance from /data/instances
"""
def get_instance(id, print_desc = True):
    path = f"/home/atg205/Documents/__Dokumente/Uni/UPMC/stage gl/DWaveDynamics2/data/instances"
    file_name = os.path.join(path, f"{id}.pckl")

    with open(file_name,'rb') as f:
        instance = pickle.load(f)
    
    if print_desc:
        print("-------")
        print(instance['about'])
        print("---------")
    return instance


"""
    Checks if H is PT symmetric
"""
def is_ptsymmetric(H):
    dim = math.log(len(H),2)
    assert dim == int(dim)
    dim = int(dim)
    P = qp.sigmax().full()
    XX = ft.reduce(np.kron, [P]*dim)
    PT_H = XX @ H.conj() @ XX
    return np.allclose(H, PT_H)


"""
    Generates a 2D PT symmetric matrix with eigenvalues +-1
"""
def generate_pt_symmetric_real_eig(a):
    assert a > 0 and a < 1, "a has to be between 0 and 1 exclusive"
    b = math.sqrt(1+a*a)
    H = np.matrix([[a*1.0j, b],[b,-a*1.0j]])
    assert is_ptsymmetric(H)
    return H

"""
    Creates a Hamiltonian of the form 
    H = |0...0><1...1| + |1...1><0...0|
"""
def create_entangled_hamiltonian(num_qubits: int):
    n = 2**num_qubits
    H = np.zeros((n, n))
    H[0, n-1] = 1
    H[n-1, 0] = 1
    return H

"""
    Create another entanglement Hamilonian for two qubits
"""
def other_entangled_hamiltonian():
    H = np.ones([4,4])
    H[0,3] = H[1,2] = H[2,1] = H[3,0] = -1.0
    print(H)
    return H

def print_graph(G):

    elarge = [(u, v) for (u, v, d) in G.edges(data=True)]

    pos = nx.spring_layout(G,k=1000, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=2)


    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def result_string_to_dict(input_string:str)->dict[int,int]:
    return {i:int(bit) for i,bit in enumerate(list(input_string)[::-1])}




