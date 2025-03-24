import numpy as np
import matplotlib.pyplot as plt

# Define the Kraus operators for the amplitude damping channel for one qubit.
def kraus_operators(gamma):
    """
    Returns the list of Kraus operators for a single qubit undergoing amplitude damping.
    gamma: damping probability (0 <= gamma <= 1)
    """
    K0 = np.array([[1, 0],
                   [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)],
                   [0, 0]])
    return [K0, K1]

# Evolve a two-qubit density matrix under independent amplitude damping channels.
def evolve_state(rho, gamma):
    """
    Evolves the density matrix 'rho' using independent amplitude damping channels on each qubit.
    gamma: damping parameter for each qubit.
    """
    Ks = kraus_operators(gamma)
    rho_new = np.zeros_like(rho, dtype=complex)
    # Use the tensor product of the Kraus operators for both qubits.
    for K1 in Ks:
        for K2 in Ks:
            K = np.kron(K1, K2)
            rho_new += K @ rho @ K.conjugate().T
    return rho_new

# Compute the concurrence of a two-qubit state.
def concurrence(rho):
    """
    Computes the concurrence of a two-qubit density matrix 'rho' using Wootters' formula.
    """
    # Pauli Y matrix
    sigma_y = np.array([[0, -1j], [1j, 0]])
    # Construct Y ⊗ Y.
    Y = np.kron(sigma_y, sigma_y)
    # Compute the "spin-flipped" state.
    rho_star = np.conjugate(rho)
    R = rho @ Y @ rho_star @ Y
    # Compute the eigenvalues and take the square roots.
    eigenvalues = np.sqrt(np.real(np.linalg.eigvals(R)))
    eigenvalues = np.sort(eigenvalues)[::-1]
    # Concurrence formula.
    C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
    return np.real(C)

# --- Main simulation ---

# Choose the parameter for the initial state.
a = 0.5
# Define the initial pure state: |psi> = sqrt(a)|01> + sqrt(1-a)|10>
psi = np.array([0, np.sqrt(a), np.sqrt(1-a), 0])
rho0 = np.outer(psi, psi.conjugate())

# Set the damping rate (kappa) and time grid.
kappa = 1 # damping rate
times = np.linspace(0,5,20)
concurrences = []

# Evolve the state over time and compute the concurrence.
for t in times:
    # Compute the damping parameter gamma(t)
    gamma = 1 - np.exp(-kappa * t)
    # Evolve the initial state under amplitude damping.
    rho_t = evolve_state(rho0, gamma)
    # Calculate the concurrence.
    C = concurrence(rho_t)
    concurrences.append(C)

# Plot the concurrence versus time.
plt.figure(figsize=(8, 6))
plt.plot(times, concurrences, label="Concurrence")
plt.xlabel("Time")
plt.ylabel("Concurrence")
plt.title("Entanglement Sudden Death: Concurrence vs. Time")
plt.grid(False)
plt.show()
