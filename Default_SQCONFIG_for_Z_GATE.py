"""
Simple configuration file for single atom qubit simulations
Just plain variable definitions - modify directly as needed
"""

import numpy as np
from pulse_functions import window_pulse
from atom_basis import make_fock_basis_states, make_initial_list_for_gate_fidelity
from qutip import sigmaz

# ============================================================================
# Time Parameters
# ============================================================================
time_unit = 1  # [μs]

# ============================================================================
# Window Pulse Parameters for delta_1(t)
# ============================================================================
delta_1_pulse_args = {"T_detuning": np.nan, "amp_delta_1": np.nan}

# ============================================================================
# Time List Parameters
# ============================================================================
scale_tlist = np.linspace(0, 1.5, 300)
# tlist = scale_tlist * T_detuning

# ============================================================================
# Hamiltonian Parameters
# ============================================================================
atom0_ham_params = {
    'Omega_01': 0,
    'delta_1': (window_pulse, delta_1_pulse_args),
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
# single atom basis ( four level : |0>, |1>, |r>, |d> )
one_atom_fock_states = make_fock_basis_states(num_qubits=1, dim_atom=4)

state0, state1, stater, stated = one_atom_fock_states
state_plus = 1 / np.sqrt(2) * (state0 + state1)
state_minus = 1 / np.sqrt(2) * (state0 - state1)

# ============================================================================
# Expectation operators ( populations or Pauli operators )
# ============================================================================
expect_list = [
    state_plus * state_plus.dag(),
    state_minus * state_minus.dag(),
    stater * stater.dag(),
    stated * stated.dag()
]

pauli_op_list = [
  state0 * state1.dag() + state1 * state0.dag(), # sigma_x
  - 1j * (state0 * state1.dag() - state1 * state0.dag()), # sigma_y
  state0 * state0.dag() - state1 * state1.dag() # sigma_z
]

# # For computations that only need computational basis
# expect_list_comp = [
#     state0 * state0.dag(),
#     state1 * state1.dag(),
# ]

# ============================================================================
# Fidelity Calculation Parameters
# ============================================================================
num_qubits = 1
dim_qubits = 2**num_qubits  # = 2
dim_atom = 4**num_qubits  # = 4
# initial state list for gate fidelity computation
psi0_list = make_initial_list_for_gate_fidelity(num_qubits, dim_atom)
qs0_list = make_initial_list_for_gate_fidelity(num_qubits, dim_qubits)
comp_indices = [0, 1]  # indices of computational basis in full Hilbert space

# target gate
target_gate = sigmaz()

# ============================================================================
# Parameter Lists for Scans
# ============================================================================



# ============================================================================
# Plotting Configuration
# ============================================================================