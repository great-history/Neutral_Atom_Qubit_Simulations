from qutip import gates, Qobj
import warnings
import numpy as np

def get_zxz_gate(alpha: float, beta: float, gamma: float):
    # Rz(alpha) Rx(beta) Rz(gamma)
    return gates.rz(alpha) * gates.rx(beta) * gates.rz(gamma)


def get_pauli_gate_embedded(pauli: str, dim: int):
    """
    Generate a Pauli gate (X, Y, Z) with zero padding for higher dimensional
    Assume the first two levels correspond to the qubit states |0> and |1>.
    """
    if dim < 2:
        raise ValueError("Dimension must be at least 2.")
    
    pad_mat = np.zeros((dim, dim), dtype=complex)
    if pauli == 'X' or pauli == 'x':
        pauli_op = np.array([[0, 1], [1, 0]])
    elif pauli == 'Y' or pauli == 'y':
        pauli_op = np.array([[0, -1j], [1j, 0]])
    elif pauli == 'Z' or pauli == 'z':
        pauli_op = np.array([[1, 0], [0, -1]])
    elif pauli == 'I' or pauli == 'i':
        pauli_op = np.array([[1, 0], [0, 1]])
    else:
        raise ValueError("Invalid Pauli operator. Choose from 'X', 'Y', or 'Z'.")
    
    pad_mat[:2, :2] = pauli_op
    return Qobj(pad_mat)


def get_CZ_gate_embedded(dim_atom: int):
    """
    Generate a CZ (Controlled-Z) gate for two qubits with zero padding.
    
    The CZ gate acts on the computational basis {|00>, |01>, |10>, |11>} as:
    |00> -> |00>
    |01> -> |01>
    |10> -> |10>
    |11> -> -|11>
    
    For higher dimensional atoms (dim_atom > 2), the gate is padded with zeros
    for states outside the computational subspace.
    
    Parameters:
    ----------
    dim_atom : int
        Dimension of each atom's Hilbert space (must be >= 2)
    
    Returns:
    -------
    CZ_gate : Qobj
        CZ gate operator with dimensions (dim_atom^2, dim_atom^2)
    
    Example:
    -------
    >>> CZ = get_CZ_gate_embedded(dim_atom=4)
    """
    if dim_atom < 2:
        raise ValueError("Dimension must be at least 2.")
    
    dim_tot = dim_atom ** 2
    cz_mat = np.zeros((dim_tot, dim_tot), dtype=complex)
    
    # The CZ gate flips the sign of |11> state, does nothing to others
    # In the full Hilbert space, |11> corresponds to index: 1 * dim_atom + 1, |00> -> 0, |01> -> 1, |10> -> dim_atom
    cz_mat[0, 0] = 1  # |00>
    cz_mat[1, 1] = 1  # |01>
    cz_mat[dim_atom, dim_atom] = 1  # |10>
    cz_mat[dim_atom + 1, dim_atom + 1] = -1 # |11>
    
    return Qobj(cz_mat, dims=[[dim_atom, dim_atom], [dim_atom, dim_atom]])
