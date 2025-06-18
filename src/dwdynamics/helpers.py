import numpy as np
import os
import pickle
import qutip as qp
import math
import networkx as nx
import matplotlib.pyplot as plt
import functools as ft
import re
import pandas as pd
from collections import defaultdict
import json
import dimod

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


def get_velox_results(system: int)->pd.DataFrame:
    df = pd.read_csv(f'../data/results/pruned/{system}/best_results_pruned_{system}_native.csv')
    df_dict= defaultdict(list)
    for row in df.itertuples():
        precision, timepoints = re.findall(r'\d+',str(row.instance))
        df_dict['precision'].append(int(precision))
        df_dict['timepoints'].append(int(timepoints))
        df_dict['num_steps'].append(int(row.num_steps))
        df_dict['runtime'].append(float(row.runtime)*1e3)
        df_dict['gap'].append(float(row.gap))
        df_dict['num_rep'].append(int(row.num_rep))
        df_dict['success_prob'].append(float(row.success_prob))
        df_dict['solution'].append(row.best_solution.replace("-1","0").replace(';',''))
        df_dict['num_var'].append(int(row.num_var))
    return pd.DataFrame(df_dict)

def get_dwave_success_rates(system: int,topology="6.4")->pd.DataFrame:
    path = f'../data/results/pruned/{system}/'

    dfs = []
    df_dict = defaultdict(list)
    for topology in [topology]:
        path += topology
        for file in os.listdir(path):
            df_dict['precision'].append(int(re.findall('(?<=precision_)\d+',file)[0]))
            df_dict['timepoints'].append(int(re.findall('(?<=timepoints_)\d+',file)[0]))
            with open(os.path.join(path,file),'r') as f:
                s = dimod.SampleSet.from_serializable(json.load(f))
        
            qpu_access_time = s.info['timing']['qpu_access_time']
            df = s.to_pandas_dataframe()
            df['energy'] = abs(round(df['energy'],10))

            df = df[['energy','num_occurrences']].groupby(by=["energy"]).sum().reset_index()
            if len(df[df.energy== 0]) == 0:
                success_rate = 0.0
            else:
                success_rate = int(df[df.energy == 0]['num_occurrences'].iloc[0])
            success_rate /= df['num_occurrences'].sum()
            
            access_time = qpu_access_time / df['num_occurrences'].sum() * 1e-3
            df_dict['topology'].append(topology)
            df_dict['success_prob'].append(success_rate)
            df_dict['runtime'].append(access_time)
            df_dict['num_var'].append(len(s.variables))
            #s['energy'] = abs(round(s['energy'],10)) 
            dfs.append(pd.DataFrame(df_dict))

    dfs_all = pd.concat(dfs)
    dfs_all = dfs_all.groupby(['precision','timepoints','topology']).mean().reset_index()

    return dfs_all

def get_precision_timepoints_pairs(dfs):
    dfs = dfs.groupby(['precision','timepoints'])['num_occurrences'].count()
    return list(set(dfs.index))

def get_velox_success_rates(system:int)->pd.DataFrame:
    df = get_velox_results(system=system)
    df = df[df.num_steps == 1000]
    df['success_prob'] = (df['success_prob'] * df['num_rep']) 
    #df['runtime'] = df['runtime'] / df['num_rep'] 

    df = df.groupby(['precision','timepoints','num_steps','num_var']).agg({'runtime': 'mean',
                                                                'num_rep' : 'sum',
                                                                'success_prob':'sum'}).reset_index()
    df['success_prob'] /= df['num_rep']
    df['runtime'] /= df['num_rep']
    df['success_prob'] /= 100

    return df

def return_tts(p_success: float,t:float, p_target=0.99)->float:
    if p_success == 0:
        return math.inf
    return (math.log(1-p_target) / math.log(1-p_success))*t
