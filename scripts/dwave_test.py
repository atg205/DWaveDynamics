import numpy as np
from dwdynamics import ComplexDynamicsProblem, Objective, helpers,instance
import matplotlib.pyplot as plt
import json
import os
import pickle
from dwave.system import DWaveSampler, EmbeddingComposite
import qutip as qp

import re
from tqdm import tqdm


def save_instance(PSI0, H, description, id, overwrite=False):
    """
    Save a quantum instance to a pickle file.

    Args:
        PSI0: Initial state.
        H: Hamiltonian.
        description (str): Description of the instance.
        id (str or int): Identifier for the instance.
        overwrite (bool): If True, overwrite existing file.
    """
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
    number_time_points = [2]
    solver_ids = ["1.4", "6.4"]
    precision = 3

    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    for solver_id in tqdm(solver_ids):
        for system in tqdm([8], leave=False):
            for ta in tqdm([10, 100, 500], leave=False):
                inst = instance.Instance(system)
                for tp in tqdm(number_time_points):
                    inst.create_instance(precision=precision, number_time_points=tp, save=False)
                    for _ in tqdm(range(5), leave=False):
                        inst.generate_and_save_sampleset(solver_id=solver_id, ta=ta)


if __name__ == "__main__":
    main()
