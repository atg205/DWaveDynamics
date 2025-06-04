import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, tensor, ket2dm, sigmax, sigmaz, sigmam, qeye, mesolve, expect, concurrence, bell_state, basis


def generate_pt_symmetric_hamiltonian(n, seed=None):
    """
    Generate a PT-symmetric Hamiltonian for an N-qubit system.

    The Hamiltonian H is defined as:
        H = 0.5*(X + P * X* * P)
    where X is a random complex matrix and the parity operator P is given by
    the tensor product of sigma_x (Pauli-X) for each qubit.

    Parameters:
        n (int): Number of qubits.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Qobj: PT-symmetric Hamiltonian (2^n x 2^n) with proper tensor structure.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Total dimension for n qubits.
    dim = 2 ** n
    # Define tensor dims for an n-qubit system.
    dims = [[2] * n, [2] * n]
    
    # Generate a random complex matrix and wrap as a Qobj with tensor dims.
    X = Qobj(np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim), dims=dims)
    
    # Build the N-qubit parity operator P = sigma_x ⊗ sigma_x ⊗ ... ⊗ sigma_x.
    P = sigmax()  # single qubit parity operator (dims: [[2], [2]])
    for _ in range(n - 1):
        P = tensor(P, sigmax())
    
    # Create the conjugate of X while preserving dims.
    X_conj = Qobj(X.full().conj(), dims=dims)
    
    # Construct the PT-symmetric Hamiltonian: H = 0.5 * (X + P * X_conj * P)
    H = 0.5 * (X + P * X_conj * P)
    return H
# # Simulation parameters
# n_qubits = 2
# dim = 2 ** n_qubits
# H = generate_pt_symmetric_hamiltonian(n_qubits, seed=1)
# print("PT-symmetric Hamiltonian (2 qubits):\n", H)

# # Define collapse (jump) operators for local decay.
# # Lowering operator sigma_minus for each qubit.
# L1 = tensor(sigmam(), qeye(2))
# L2 = tensor(qeye(2), sigmam())
# c_ops = [L1, L2]

# # Define the initial Bell state |Φ⁺⟩ = 1/√2 (|00⟩ + |11⟩)
# psi0 = tensor(basis(2,0),basis(2,1)) #
# #psi0=bell_state(state='00')

# rho0 = ket2dm(psi0)

# # Define the two-qubit correlation operator: sigma_z ⊗ sigma_z
# corr_op = tensor(sigmaz(), sigmaz())

# # Time parameters for the simulation
# t_min = 0.0
# t_max = 10.0
# num_points = 200
# times = np.linspace(t_min, t_max, num_points)

# # Solve the master equation using mesolve
# result = mesolve(H, rho0, times, c_ops, [])

# # Prepare lists to store concurrence and correlation values.
# concurrences = []
# correlations = []

# # Loop over the time-evolved density matrices
# for rho in result.states:
#     # Compute concurrence (QuTiP provides a concurrence function for two-qubit states)
#     C = concurrence(rho)
#     concurrences.append(C)
#     # Compute the expectation value of sigma_z ⊗ sigma_z
#     correlations.append(expect(corr_op, rho))

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(times, concurrences, label='Concurrence', linewidth=2)
# plt.plot(times, correlations, label='Correlation ⟨σ_z ⊗ σ_z⟩', linewidth=2)
# plt.xlabel('Time', fontsize=14)
# #plt.ylabel('Value', fontsize=14)
# plt.title('Pt-symmetric dynamics', fontsize=16)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.savefig("pt_symmetry.pdf")
# plt.show()
