import numpy as np
import os
import pickle

def random_matrix(dims, hermitian=True):
    A = np.random.uniform(-1, 1, dims) + 1.j * np.random.uniform(-1, 1, dims)
    if hermitian:
        return A + A.conj().T
    return A


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