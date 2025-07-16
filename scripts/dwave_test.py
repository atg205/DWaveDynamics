import numpy as np
from dwdynamics import ComplexDynamicsProblem, Objective, helpers,instance # Difference a. We are using ComplexDynamicsProblem
import matplotlib.pyplot as plt
import json
import os
import pickle
from dwave.system import DWaveSampler, EmbeddingComposite
import qutip as qp

import re
from tqdm import tqdm


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

def main():
    number_time_points = [i for i in range(2,5)]
    number_time_points = [2]
    solver_id = "6.4"
    precision =2


    sigma_x = np.array([[0,1],[1,0]])
    sigma_y = np.array([[0,-1j],[1j,0]])
    sigma_z = np.array([[1,0],[0,-1]])




    for system in tqdm([9]):
        for ta in tqdm([200],leave=False):
            inst = instance.Instance(system)
            for tp in tqdm(number_time_points):
                inst.create_instance(precision=precision, number_time_points=tp,save=False)
                for _ in tqdm(range(5),leave=False):
                    inst.generate_and_save_sampleset(solver_id=solver_id,ta=ta)





if __name__ == "__main__":
    main()
