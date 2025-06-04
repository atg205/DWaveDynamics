from dwdynamics import ComplexDynamicsProblem,RealDynamicsProblem, Objective
from dimod import ExactSolver
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm

def random_matrix(dims, hermitian=True):
    A = np.random.uniform(-1, 1, dims) + 1.j * np.random.uniform(-1, 1, dims)
    if hermitian:
        return A + A.conj().T
    return A


"""
 Test for random number of instances that the lowest ground state energy decreases with increasing precision
"""
def main():
    exact_solver = ExactSolver()
    
    number_time_points = 2
    PSI0 = np.array([1, 0], dtype=np.complex128)
    results = []
    for _ in tqdm(range(10)):
        H = random_matrix([2,2])
        exact_results = []
        for precision in [2,3]:
            # generate qubo

            problem = ComplexDynamicsProblem(
                hamiltonian=H,      
                initial_state = PSI0,
                times=tuple(range(number_time_points)),             
                num_bits_per_var=precision                
            )
            qubo = problem.qubo(objective=Objective.norm)  # Other choice would be to use Objective.hessian                
            assert qubo.num_variables == problem.hamiltonian.shape[0] * len(problem.times) * problem.num_bits_per_var * 2

            # solve numerically
            exact_results.append(exact_solver.sample(qubo).first.energy)
            print(qubo.num_variables)
        #assert(exact_results[0] > exact_results[1])
        results.append(exact_results[0] / exact_results[1])
    print(results)
            
        

if __name__ == "__main__":
    main()