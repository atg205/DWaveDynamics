import numpy as np
import networkx as nx
from dwdynamics import ComplexDynamicsProblem, Objective, helpers # Difference a. We are using ComplexDynamicsProblem
import matplotlib.pyplot as plt
import json
import os
from dwave_networkx.generators.pegasus import (get_tuple_defragmentation_fn, fragmented_edges,
    pegasus_coordinates, pegasus_graph)
from dwave.system import DWaveSampler, EmbeddingComposite
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from tqdm import tqdm
import ptsymmetric
import pickle

"""
    Saves the information concerning an instance to a json file in /data/instance
"""
def save_instance(PSI0, H, description, id, overwrite=False):
    instance_dict = {
        'psi0': PSI0,
        'H': H,
        'about': description
    }
    path = f"data/instances/"
    file_name = os.path.join(path, f"{id}.pckl")
    with open(file_name,'wb' if overwrite else 'xb') as f:
        pickle.dump(instance_dict,f)

"""
    Creates an instance based on the provided parameters and saves the qubo to /data/instances
"""
def create_instance(instance_id,precision, number_time_points):
    instance = helpers.get_instance(instance_id)

    problem = ComplexDynamicsProblem(
        hamiltonian=instance['H'],     
        initial_state = instance['psi0'],            
        times=tuple(range(number_time_points)),             
        num_bits_per_var=precision                
    )
    qubo = problem.qubo(objective=Objective.norm)  # Other choice would be to use Objective.hessian            
    assert qubo.num_variables == problem.hamiltonian.shape[0] * len(problem.times) * problem.num_bits_per_var * 2

    # save instances in the form 
    # systemid_{d}_precision_{d}_timepoints_{d}.json
    path = f"data/instances/pruned/{instance_id}"

    file_name = os.path.join(path, f"precision_{precision}_timepoints_{number_time_points}.json")
    os.makedirs(path, exist_ok=True)
    with open(file_name,'w') as f:
        json.dump(qubo.to_serializable(),f)
    return qubo

"""
    Runs one sample on the indicated D-Wave machine
"""
def get_sampleset(qubo,solver_id="5.4"):

    if solver_id == "5.4":
        dw_sampler = EmbeddingComposite(DWaveSampler( solver="Advantage_system5.4", region="eu-central-1", ))
    elif solver_id == "2.6": # zephyr
        dw_sampler = EmbeddingComposite(DWaveSampler( solver="Advantage2_prototype2.6"))
    elif solver_id == "6.4": 
        dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage_system6.4"))
    else:
        raise ValueError("Invalid solver id")

    dw_result = dw_sampler.sample(qubo, num_reads=1000, annealing_time=200)

    return dw_result

"""
    Returns last index of result files so files are not overwritten 
"""
def get_last_index(files):
    if files == []:
        return 0
    return max([int(re.findall('\d+(?=\.json)',file)[0]) for file in files])


"""
    Saves D-Wave results to /data/results
"""
def save_dwave_result(dw_result, instance_id, solver_id, precision, timepoints):
    path = f"data/results/pruned/{instance_id}/{solver_id}"
    os.makedirs(path, exist_ok=True)
    idx = get_last_index(os.listdir(path)) +1

    file_name = os.path.join(path, f"precision_{precision}_timepoints_{timepoints}_{idx}.json")

    with open(file_name,'w') as f:
        json.dump(dw_result.to_serializable(),f)

def create_entangled_hamiltonian(num_qubits):
    n = 2**num_qubits
    H = np.zeros((n, n))
    H[0, n-1] = 1
    H[n-1, 0] = 1
    return H

def main():
    precision = 2
    number_time_points = 2
    instance_id = 6
    solver_id = "5.4"


    PSI0 = np.array([1, 0,0,0,0,0,0,0], dtype=np.complex128)  
    H = np.pi/2*create_entangled_hamiltonian(3)
    print(H)
    #save_instance(PSI0,H,"two qubit entanglement", instance_id)

    qubo = create_instance(precision=precision, number_time_points=number_time_points, instance_id=instance_id)

    for _ in tqdm(range(10)):
        dw_result = get_sampleset(qubo, solver_id) 
        save_dwave_result(dw_result,instance_id=instance_id, solver_id=solver_id, precision=precision,timepoints=number_time_points)



if __name__ == "__main__":
    main()
