import numpy as np
from dwdynamics import ComplexDynamicsProblem, Objective, helpers,instance # Difference a. We are using ComplexDynamicsProblem
import matplotlib.pyplot as plt
import json
import os
import pickle
from dwave.system import DWaveSampler, EmbeddingComposite


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
    number_time_points = [2,3,4]
    instance_id = 9
    solver_id = "5.4"
    precision = [2,3,4,5,6,7]
    
    for tp in number_time_points:
        for p in precision:
            inst = instance.Instance(instance_id)
            inst.create_instance(p,tp,save=True)
    


    
        


if __name__ == "__main__":
    main()
