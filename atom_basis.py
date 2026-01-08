"""
Basis state generation functions for neutral atom qubits
"""

from qutip import Qobj, basis, tensor
import numpy as np
import itertools


def make_fock_basis_states(num_qubits: int, dim_atom: int):
    """
    Generate all Fock basis states for a multi-qubit system in a multi-dimensional list.
    
    Parameters:
    ----------
    num_qubits : int
        Number of qubits
    dim_atom : int
        Dimension of each atom's Hilbert space
    
    Returns:
    -------
    basis_states : nested list of Qobj
        Multi-dimensional list of shape (dim_atom, dim_atom, ..., dim_atom) with num_qubits dimensions
        e.g., for 2 qubits and dim_atom=3: 
        [
          [|00>, |01>, |02>],
          [|10>, |11>, |12>],
          [|20>, |21>, |22>]
        ]
        Access via basis_states[i][j][k]... for state |ijk...>
    """
    # First generate all states in a flat list
    flat_list = []
    for indices in itertools.product(range(dim_atom), repeat=num_qubits):
        # Create tensor product of single-atom basis states
        cur_ket = basis(dim_atom, indices[0])
        for qIdx in range(1, num_qubits):
            cur_ket = tensor(cur_ket, basis(dim_atom, indices[qIdx]))
        flat_list.append(cur_ket)
    
    # Reshape into multi-dimensional nested list
    shape = [dim_atom] * num_qubits
    basis_states = np.array(flat_list, dtype=object).reshape(shape).tolist()
    
    return basis_states


def make_computational_basis_states(num_qubits: int, dim_atom: int):
    """
    Generate all computational basis states for a multi-qubit system in a multi-dimensional list.
    
    Parameters:
    ----------
    num_qubits : int
        Number of qubits
    dim_atom : int
        Dimension of each atom's Hilbert space (>= 2)
    
    Returns:
    -------
    basis_states : nested list of Qobj
        Multi-dimensional list of shape (2, 2, ..., 2) with num_qubits dimensions
        e.g., for 2 qubits: 
        [
          [|00>, |01>],
          [|10>, |11>]
        ]
        Access via basis_states[i][j][k]... for state |ijk...> where i,j,k ∈ {0,1}
    """
    # First generate all states in a flat list
    flat_list = []
    for bits in itertools.product([0, 1], repeat=num_qubits):
        # Create tensor product of single-qubit basis states
        cur_ket = basis(dim_atom, bits[0])
        for qIdx in range(1, num_qubits):
            cur_ket = tensor(cur_ket, basis(dim_atom, bits[qIdx]))
        flat_list.append(cur_ket)
    
    # Reshape into multi-dimensional nested list
    shape = [2] * num_qubits
    basis_states = np.array(flat_list, dtype=object).reshape(shape).tolist()
    
    return basis_states


def make_superposition_state(num_qubits: int, dim_atom: int):
    """
    Generate equal superposition state: |ψ> = (1/√d) Σ|i> over all computational basis states.
    
    Parameters:
    ----------
    num_qubits : int
        Number of qubits
    dim_atom : int
        Dimension of each atom's Hilbert space
    
    Returns:
    -------
    superposition_ket : Qobj
        Equal superposition of all computational basis states
    """
    dim_tot = dim_atom ** num_qubits
    dim_qubit = 2 ** num_qubits
    
    # Start with zero state
    ket_super = Qobj(arg=np.zeros((dim_tot, 1)), dims=[[dim_atom]*num_qubits, [1]])
    
    # Sum over all computational basis states
    for bits in itertools.product([0, 1], repeat=num_qubits):
        cur_ket = basis(dim_atom, bits[0])
        for qIdx in range(1, num_qubits):
            cur_ket = tensor(cur_ket, basis(dim_atom, bits[qIdx]))
        ket_super = ket_super + cur_ket
    
    # Normalize
    ket_super = (1 / np.sqrt(dim_qubit)) * ket_super
    
    return ket_super


def make_initial_list_for_gate_fidelity(num_qubits: int, dim_atom: int):
    """
    Create the 2^num_qubits + 1 states used for computing gate fidelities.
    
    Includes:
      - All 2^num_qubits computational basis states
      - 1 equal superposition state
    
    Parameters:
    ----------
    num_qubits : int
        Number of qubits
    dim_atom : int
        Dimension of each atom's Hilbert space (>= 2)
    
    Returns:
    -------
    state_list : list[Qobj]
        List of ket states
    dm_list : list[Qobj]
        List of corresponding density matrices
    """
    state_list = []
    
    # Generate all computational basis states (returns multi-dimensional list)
    basis_states = make_computational_basis_states(num_qubits, dim_atom)
    
    # Flatten the multi-dimensional list to iterate over all basis states
    flat_basis_states = np.array(basis_states, dtype=object).flatten()
    for ket in flat_basis_states:
        state_list.append(ket)
    
    # Add equal superposition state
    superposition_ket = make_superposition_state(num_qubits, dim_atom)
    state_list.append(superposition_ket)
    
    # # Density matrix for maximally mixed state over computational subspace
    # dm_tr = superposition_ket * superposition_ket.dag()
    
    return state_list


def make_single_basis_state(qubit_values: tuple, dim_atom: int):
    """
    Generate a specific computational basis state.
    
    Parameters:
    ----------
    qubit_values : tuple of int
        Values for each qubit (0 or 1), e.g., (0, 1, 1) for |011>
    dim_atom : int
        Dimension of each atom's Hilbert space
    
    Returns:
    -------
    ket : Qobj
        The requested basis state
    
    Example:
    -------
    >>> state_011 = make_single_basis_state((0, 1, 1), dim_atom=2)
    """
    num_qubits = len(qubit_values)
    ket = basis(dim_atom, qubit_values[0])
    for qIdx in range(1, num_qubits):
        ket = tensor(ket, basis(dim_atom, qubit_values[qIdx]))
    return ket


def make_custom_superposition(coefficients: list, num_qubits: int, dim_atom: int):
    """
    Generate a custom superposition state with given coefficients.
    
    Parameters:
    ----------
    coefficients : list of complex
        Coefficients for each computational basis state
        Length must be 2^num_qubits
    num_qubits : int
        Number of qubits
    dim_atom : int
        Dimension of each atom's Hilbert space
    
    Returns:
    -------
    ket : Qobj
        The custom superposition state (automatically normalized)
    
    Example:
    -------
    >>> # Create (|00> + i|11>) state for 2 qubits
    >>> state = make_custom_superposition([1, 0, 0, 1j], num_qubits=2, dim_atom=2)
    """
    dim_tot = dim_atom ** num_qubits
    dim_qubit = 2 ** num_qubits
    
    if len(coefficients) != dim_qubit:
        raise ValueError(f"Expected {dim_qubit} coefficients for {num_qubits} qubits")
    
    ket = Qobj(arg=np.zeros((dim_tot, 1)), dims=[[dim_atom]*num_qubits, [1]])
    
    basis_states = make_computational_basis_states(num_qubits, dim_atom)
    for i, coeff in enumerate(coefficients):
        ket = ket + coeff * basis_states[i]
    
    # Normalize
    norm = ket.norm()
    if norm > 0:
        ket = ket / norm
    
    return ket


# # single atom basis ( four level : |0>, |1>, |r>, |d> )
# one_atom_fock_states = make_fock_basis_states(num_qubits=1, dim_atom=4)
# # two (double) atom basis |i,j> ( i,j = 0,1,r,d )
# two_atom_fock_states = make_fock_basis_states(num_qubits=2, dim_atom=4)