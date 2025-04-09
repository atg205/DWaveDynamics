import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, basis, sigmam, qeye, mesolve, ket2dm, Qobj, sigmax, sigmay, sigmaz, bell_state, w_state, ghz_state

# Function to compute concurrence of a two-qubit density matrix.
def concurrence(rho):
    """
    Calculate the concurrence of a two-qubit state rho.
    The formula uses the eigenvalues of R = rho (σ_y⊗σ_y) rho* (σ_y⊗σ_y),
    where sigma_y is the Pauli Y matrix.
    """
    # Define Pauli Y operator
    sy = Qobj([[0, -1j], [1j, 0]])
    # Build sigma_y tensor sigma_y
    Y = tensor(sy, sy)
    
    # Compute the "spin-flipped" state
    R = (rho * Y * rho.conj() * Y).full()
    
    # Compute eigenvalues of R
    eigenvalues = np.linalg.eigvals(R)
    # Take square roots of the absolute values of the eigenvalues
    sqrt_eig = np.sqrt(np.abs(eigenvalues))
    # Sort in increasing order
    sqrt_eig = np.sort(sqrt_eig)
    
    # Concurrence: max(0, λ1 - λ2 - λ3 - λ4) where λ1 is the largest eigenvalue.
    conc = max(0, sqrt_eig[-1] - sqrt_eig[-2] - sqrt_eig[-3] - sqrt_eig[-4])
    return np.real(conc)

# System parameters
gamma = 1  # amplitude damping rate
tlist = np.linspace(0, 5, 100)  # time array

# Collapse operators for amplitude damping on each qubit:
c1 = np.sqrt(gamma) * tensor(sigmam(), qeye(2))
c2 = np.sqrt(gamma) * tensor(qeye(2), sigmam())
c_ops = [c1, c2]

# Initial state: maximally entangled state
#psi0 =  w_state(2) #w state
psi0= ghz_state(2) #ghz state
#psi0= bell_state(state='00') #bell state
rho0 = ket2dm(psi0)

# Hamiltonian is set to zero (purely dissipative dynamics)
H = tensor(sigmaz(),sigmaz()) + tensor(sigmax(),sigmax())# + tensor(sigmax(),sigmax())

# Solve the master equation using QuTiP's mesolve
result = mesolve(H, rho0, tlist, c_ops, [])

# Compute the concurrence at each time point
concurrences = [concurrence(rho) for rho in result.states]

# Define Pauli operators for computing correlation functions.
sz = sigmaz()
sx = sigmax()
sy = sigmay()

# Compute the correlation functions for each time point.
corr_zz = [np.real((rho * tensor(sz, sz)).tr()) for rho in result.states]
corr_xx = [np.real((rho * tensor(sx, sx)).tr()) for rho in result.states]
corr_yy = [np.real((rho * tensor(sy, sy)).tr()) for rho in result.states]

# Plotting: Create subplots for concurrence and correlation functions.
fig, ax = plt.subplots(2, 1, figsize=(8, 10))

# Plot concurrence
ax[0].plot(tlist, concurrences, 'b-', linewidth=2)
ax[0].set_xlabel('Time', fontsize=14)
ax[0].set_ylabel('Concurrence', fontsize=14)
#ax[0].set_title('Entanglement Sudden Death', fontsize=16)

# Plot correlation functions
ax[1].plot(tlist, corr_zz, 'r-', label=r'$\langle \sigma_z \otimes \sigma_z \rangle$', linewidth=2)
ax[1].plot(tlist, corr_xx, 'g-', label=r'$\langle \sigma_x \otimes \sigma_x \rangle$', linewidth=2)
ax[1].plot(tlist, corr_yy, 'm-', label=r'$\langle \sigma_y \otimes \sigma_y \rangle$', linewidth=2)
ax[1].set_xlabel('Time', fontsize=14)
ax[1].set_ylabel('Correlation', fontsize=14)
ax[1].set_title('Two-Qubit Correlation Functions', fontsize=16)
ax[1].legend(fontsize=12)

plt.tight_layout()
plt.savefig("entanglement_SD.pdf")
plt.show()
