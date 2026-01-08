from qutip import basis, tensor, qeye, coefficient
import numpy as np
from pulse_functions import *
import warnings

"""
  Constructs the Hamiltonian for a single-atom system ( time-independent ).

  Returns:
  - Qobj: The total Hamiltonian operator for the single-atom system.
"""
def construct_single_atom_hamiltonian(ham_params, lindblad_params):
  # Basis states for a single atom
  state0 = basis(4, 0)
  state1 = basis(4, 1)
  stater = basis(4, 2)
  # stated = basis(4, 3)

  # Hamiltonian for atom 0
  H0 = ((ham_params['Omega_01'] / 2.0) * (state0 * state1.dag() + state1 * state0.dag())
      + (ham_params['delta_1']) * (state1 * state1.dag())
      + (ham_params['Omega_r'] / 2.0) * (state1 * stater.dag() + stater * state1.dag())
      + (ham_params['Delta_r']) * (stater * stater.dag())
  )
  H0 = tensor(H0, qeye(4))  # Extend to two-atom space

  # Collapse operators: r -> {0,1,d} or vice versa
  collapse_list = [
      np.sqrt(lindblad_params['b_0r'] * lindblad_params['gamma_r']) * state0 * stater.dag(),
      np.sqrt(lindblad_params['b_1r'] * lindblad_params['gamma_r']) * state1 * stater.dag(),
      np.sqrt(lindblad_params['b_dr'] * lindblad_params['gamma_r']) * stated * stater.dag(),
  ]

  return H0, collapse_list

"""
  Constructs the Hamiltonian for a two-atom system with Rydberg blockade interaction. ( time-independent )

  This function builds the total Hamiltonian by combining individual Hamiltonians for each atom
  and adding a coupling term representing the Rydberg blockade effect between the two atoms.
  The system uses a 4-level basis per atom: |0>, |1>, |r> (Rydberg), |d> (other states).

  Returns:
  - Qobj: The total Hamiltonian operator for the two-atom system.
"""
def construct_two_atom_hamiltonian(atom0_ham_params, atom1_ham_params, lindblad_params, Rydberg_B):
  # Basis states for a single atom
  state0 = basis(4, 0)
  state1 = basis(4, 1)
  stater = basis(4, 2)
  stated = basis(4, 3)
  
  # Hamiltonian for atom 0
  H0 = ((atom0_ham_params['Omega_01'] / 2.0) * (state0 * state1.dag() + state1 * state0.dag())
      + (atom0_ham_params['delta_1']) * (state1 * state1.dag())
      + (atom0_ham_params['Omega_r'] / 2.0) * (state1 * stater.dag() + stater * state1.dag())
      + (atom0_ham_params['Delta_r']) * (stater * stater.dag())
  )
  H0 = tensor(H0, qeye(4))  # Extend to two-atom space

  # Hamiltonian for atom 1
  H1 = ((atom1_ham_params['Omega_01'] / 2.0) * (state0 * state1.dag() + state1 * state0.dag())
      + (atom1_ham_params['delta_1']) * (state1 * state1.dag())
      + (atom1_ham_params['Omega_r'] / 2.0) * (state1 * stater.dag() + stater * state1.dag())
      + (atom1_ham_params['Delta_r']) * (stater * stater.dag())
  )
  H1 = tensor(qeye(4), H1)  # Extend to two-atom space

  H_coupling = Rydberg_B * (tensor(stater * stater.dag(), stater * stater.dag()))

  # Collapse operators: r -> {0,1,d} or vice versa
  c_ops_list = [
      np.sqrt(lindblad_params['b_0r'] * lindblad_params['gamma_r']) * state0 * stater.dag(),
      np.sqrt(lindblad_params['b_1r'] * lindblad_params['gamma_r']) * state1 * stater.dag(),
      np.sqrt(lindblad_params['b_dr'] * lindblad_params['gamma_r']) * stated * stater.dag(),
  ]
  collapse_list = [tensor(c_op, qeye(4)) for c_op in c_ops_list] \
            + [tensor(qeye(4), c_op) for c_op in c_ops_list]

  return H0 + H1 + H_coupling, collapse_list


"""
  Constructs the Hamiltonian for a single-atom system ( time-dependent ).

  Returns:
  - Qobj: The total Hamiltonian operator for the single-atom system.
"""
def get_param(param_dict, key):
  val = param_dict[key]
  if isinstance(val, tuple) and len(val) == 2:
      func, my_args = val
      return coefficient(func, args=my_args)
  elif isinstance(val, (int, float)):
    return val
  else:
    warnings.warn(f"Unexpected value type for key '{key}': {type(val)}. Stopping execution.", UserWarning)

"""
  Constructs the time-dependent Hamiltonian for a single-atom system.

  Parameters:
  - ham_params: dict with Hamiltonian parameters (constants or (func, args) tuples)
  - lindblad_params: dict with Lindblad parameters

  Returns:
  - tuple: (H0, collapse_list) where H0 is the Hamiltonian, collapse_list is the list of collapse operators
"""
def construct_TD_SAHam(ham_params: dict, lindblad_params: dict):
  # Basis states for a single atom
  state0 = basis(4, 0)
  state1 = basis(4, 1)
  stater = basis(4, 2)
  stated = basis(4, 3)
  
  H0_0 = (1 / 2.0) * (state0 * state1.dag() + state1 * state0.dag()) \
        * get_param(ham_params, "Omega_01")
  H0_1 = (state1 * state1.dag()) * get_param(ham_params, "delta_1")
  H0_2 = (1 / 2.0) * (state1 * stater.dag() + stater * state1.dag()) \
        * get_param(ham_params, "Omega_r")
  H0_3 = (stater * stater.dag()) * get_param(ham_params, "Delta_r")
  H0 = H0_0 + H0_1 + H0_2 + H0_3

  # Collapse operators: r -> {0,1,d} or vice versa
  collapse_list = [
      np.sqrt(lindblad_params['b_0r'] * lindblad_params['gamma_r']) * state0 * stater.dag(),
      np.sqrt(lindblad_params['b_1r'] * lindblad_params['gamma_r']) * state1 * stater.dag(),
      np.sqrt(lindblad_params['b_dr'] * lindblad_params['gamma_r']) * stated * stater.dag(),
  ]

  return H0, collapse_list


def construct_TD_TAHam(atom0_ham_params, atom1_ham_params, \
                       atom0_lindblad_params, atom1_lindblad_params, Rydberg_B):
  # Basis states for a single atom
  state0 = basis(4, 0)
  state1 = basis(4, 1)
  stater = basis(4, 2)
  stated = basis(4, 3)
  
  # Hamiltonian for atom 0
  H0 = (1 / 2.0) * (state0 * state1.dag() + state1 * state0.dag()) * get_param(atom0_ham_params, "Omega_01") \
      + (state1 * state1.dag()) * get_param(atom0_ham_params, "delta_1") \
      + (1 / 2.0) * (state1 * stater.dag() + stater * state1.dag()) * get_param(atom0_ham_params, "Omega_r") \
      + (stater * stater.dag()) * get_param(atom0_ham_params, "Delta_r")
  H0 = tensor(H0, qeye(4))  # Extend to two-atom space

  # Hamiltonian for atom 1 
  H1 = (1 / 2.0) * (state0 * state1.dag() + state1 * state0.dag()) * get_param(atom1_ham_params, "Omega_01") \
      + (state1 * state1.dag()) * get_param(atom1_ham_params, "delta_1") \
      + (1 / 2.0) * (state1 * stater.dag() + stater * state1.dag()) * get_param(atom1_ham_params, "Omega_r") \
      + (stater * stater.dag()) * get_param(atom1_ham_params, "Delta_r")
  H1 = tensor(qeye(4), H1)  # Extend to two-atom space

  # coupling term
  H_coupling = Rydberg_B * (tensor(stater * stater.dag(), stater * stater.dag()))

  # Collapse operators: r -> {0,1,d} or vice versa
  c0_ops_list = [
      np.sqrt(atom0_lindblad_params['b_0r'] * atom0_lindblad_params['gamma_r']) * state0 * stater.dag(),
      np.sqrt(atom0_lindblad_params['b_1r'] * atom0_lindblad_params['gamma_r']) * state1 * stater.dag(),
      np.sqrt(atom0_lindblad_params['b_dr'] * atom0_lindblad_params['gamma_r']) * stated * stater.dag(),
  ]
  c1_ops_list = [
      np.sqrt(atom1_lindblad_params['b_0r'] * atom1_lindblad_params['gamma_r']) * state0 * stater.dag(),
      np.sqrt(atom1_lindblad_params['b_1r'] * atom1_lindblad_params['gamma_r']) * state1 * stater.dag(),
      np.sqrt(atom1_lindblad_params['b_dr'] * atom1_lindblad_params['gamma_r']) * stated * stater.dag(),
  ]

  collapse_list = [tensor(c_op, qeye(4)) for c_op in c0_ops_list] \
            + [tensor(qeye(4), c_op) for c_op in c1_ops_list]

  return H0 + H1 + H_coupling, collapse_list