"""
Simple configuration file for single atom qubit simulations
Just plain variable definitions - modify directly as needed
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import myPkg
try:
    parent_dir = Path(__file__).resolve().parent.parent
except NameError:
    # When __file__ is not available (e.g., in interactive mode)
    parent_dir = Path.cwd()
    
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
from common_imports import *
from qutip import sigmax

# ============================================================================
# Time Parameters
# ============================================================================
time_unit = 1  # [μs]

# ============================================================================
# Gaussian Pulse Parameters
# ============================================================================
sigma = 1.0 * time_unit  # pulse width [μs]
scale_t0 = 10  # multiplier for pulse center
t0 = scale_t0 * sigma  # pulse center
Omega_01 = 1.0  # Rabi frequency [MHz]

# Pulse arguments dictionary
Omega_01_pulse_args = {
    "sigma": sigma,
    "t0": t0,
    "amp_Omega_01": Omega_01
}

# ============================================================================
# Time List Parameters
# ============================================================================
num_widths = 30
num_points_per_width = 3
scale_tlist = np.linspace(0, num_widths, num_widths * num_points_per_width)
tlist = scale_tlist * sigma

# Scaled Gaussian pulse shape (for plotting)
scale_pulse_shape = gaussian_pulse(scale_tlist, sigma=1, t0=scale_t0, amp_Omega_01=Omega_01)

# ============================================================================
# Hamiltonian Parameters
# ============================================================================
atom0_ham_params = {
    'Omega_01': (gaussian_pulse, Omega_01_pulse_args),
    'delta_1': 0,
    'Omega_r': 0,
    'Delta_r': 0
}

# ============================================================================
# Lindblad Parameters
# ============================================================================
lindblad_params = {
    'gamma_r': 1 / 540,  # decay rate (set to 0 for no decay)
    'b_0r': 1/16,
    'b_1r': 1/16,
    'b_dr': 7/8
}

# ============================================================================
# Basis States
# ============================================================================
state0, state1, stater, stated = make_fock_basis_states(num_qubits=1, dim_atom=4)

# Expectation operators
expect_list = [
    state0 * state0.dag(),
    state1 * state1.dag(),
    stated * stated.dag(),
    stater * stater.dag(),
]

# For computations that only need computational basis
expect_list_comp = [
    state0 * state0.dag(),
    state1 * state1.dag(),
]

# ============================================================================
# Fidelity Calculation Parameters
# ============================================================================
num_qubits = 1
dim_qubits = 2**num_qubits  # = 2
dim_atom = 4**num_qubits  # = 4
comp_indices = [0, 1]  # indices of computational basis in full Hilbert space
target_gate = sigmax()

# ============================================================================
# Common Parameter Lists for Scans
# ============================================================================

# For scanning Omega_01 (with fixed sigma)
Omega_01_list = np.linspace(0, 1.75, 8).tolist()

# For scanning sigma (with fixed Omega_01)
sigma_list = np.array([0.5, 0.75, 1, 1.25, 1.5, 2, 4, 8]) * time_unit

# ============================================================================
# Plotting Configuration
# ============================================================================

# Global pulse dictionary for plotting
global_pulse_dict = {
    'gaussian_pulse': {
        'data': scale_pulse_shape,  
        'label': 'Gaussian', 
        'style': 'Gaussian_pulse'
    }
}