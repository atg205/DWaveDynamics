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
    number_time_points = [2,3,4,5]
    solver_id = "6.4"
    systems = [1,2,4,5,6,7]
    systems = [11]
    precision =3

    for system in [16]:
        systems = [system]
        N = 2
        psi0,H = helpers.create_entangled_parity_hamiltonian(N)
        save_instance(psi0,0.5*H,f"0.5* entangled parity hamiltonian with {N} qubits",system,overwrite=True)

        for system in tqdm(systems):
            inst = instance.Instance(system)
            for tp in tqdm(number_time_points):
                inst.create_instance(precision=precision, number_time_points=tp,save=True)
                for _ in tqdm(range(5)):
                    inst.generate_and_save_sampleset(solver_id="6.4")





if __name__ == "__main__":
    main()
