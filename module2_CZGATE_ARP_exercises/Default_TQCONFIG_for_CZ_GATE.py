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
from qutip import Qobj

# ============================================================================
# Time Parameters
# ============================================================================
time_unit = 1  # [μs]

# ============================================================================
# Adiabatic Rapid Passage Pulse Parameters
# ============================================================================
T_gate = 0.54  # [us] 
tau = 0.175 * T_gate  # [us] we can chose it as a time unit

Omega_r_pulse_args = dict(
  amp_Omega_r= 2 * np.pi * 17, 
  T_gate = T_gate, 
  tau = tau
)  # [MHz, us, us]

Delta_r_pulse_args = dict(
  amp_Delta_r= 2 * np.pi * 23, 
  T_gate = T_gate, 
  tau = tau
)  # [MHz, us, us]

# ============================================================================
# Time List Parameters
# ============================================================================
num_tpoints = 300 + 1
scale_tlist = np.linspace(0, 1, num_tpoints) * time_unit  # [us]
tlist = scale_tlist * T_gate  # [us]

# ============================================================================
# Hamiltonian Parameters
# ============================================================================
atom0_ham_params = dict(
  Omega_01= 0, 
  delta_1= 0, 
  Omega_r= (APR_pulse_Omega_r, Omega_r_pulse_args),  # Tuple: (function, args)
  Delta_r= (APR_pulse_Delta_r, Delta_r_pulse_args) # Tuple: (function, args)
)

atom1_ham_params = dict(
  Omega_01 = 0, 
  delta_1 = 0, 
  Omega_r = (APR_pulse_Omega_r, Omega_r_pulse_args), # Tuple: (function, args)
  Delta_r = (APR_pulse_Delta_r, Delta_r_pulse_args) # Tuple: (function, args)
)

# ============================================================================
# Lindblad Parameters
# ============================================================================
lindblad_params = dict(
  gamma_r = 1 / 540, 
  b_0r = 1/16, 
  b_1r = 1/16, 
  b_dr = 7/8
)

# ============================================================================
# Coupling Parameters
# ============================================================================
Rydberg_B = 2 * np.pi * 200 # [MHz]

# ============================================================================
# Basis States
# ============================================================================
# single atom basis ( four level : |0>, |1>, |r>, |d> )
state0, state1, stater, stated = make_fock_basis_states(num_qubits=1, dim_atom=4)
# two (double) atom basis |i,j> ( i,j = 0,1,r,d )
two_atom_fock_states = make_fock_basis_states(num_qubits=2, dim_atom=4)


# ============================================================================
# list of wanted operators
# ============================================================================
## set I
# pop_stated = tensor(stated * stated.dag(), Qobj(np.eye(4)))  # |d><d| ⊗ I
# state_1rr1 = 1 / np.sqrt(2) * (two_atom_fock_states[1][2] + two_atom_fock_states[2][1]) # |1,r> + |r,1> normalized

# expect_list = [
#   state_1rr1 * state_1rr1.dag(),
#   two_atom_fock_states[2][0] * two_atom_fock_states[2][0].dag(),
#   two_atom_fock_states[2][2] * two_atom_fock_states[2][2].dag(),
#   pop_stated,
# ]

## set II
# pop_stated = tensor(stated * stated.dag(), Qobj(np.eye(4)))  # |d><d| ⊗ I

# expect_op_list = [
#   psi0 * psi0.dag(),
#   two_atom_fock_states[2][0] * two_atom_fock_states[2][0].dag(),
#   pop_stated,
# ]

# ============================================================================
# Fidelity Calculation Parameters
# ============================================================================
num_qubits = 2
dim_qubits = 2**num_qubits  # = 4
dim_atom = 4**num_qubits  # = 16
comp_indices = [0, 1, 4, 5]  # indices of computational basis in full Hilbert space
# initial state list for gate fidelity computation
# psi0_list = make_initial_list_for_gate_fidelity(num_qubits = 2, dim_atom = 4)
# qs0_list = make_initial_list_for_gate_fidelity(num_qubits = 2, dim_atom = 2)

target_gate = np.diag([1, -1, -1, -1])  # CZ gate in the two-qubit computational basis
target_gate = Qobj(target_gate, dims=[[2, 2], [2, 2]])  # target gate in the two-qubit qubit subspace

# ============================================================================
# Common Parameter Lists for Scans
# ============================================================================

# ============================================================================
# Plotting Configuration
# ============================================================================