from qutip import Qobj, basis, tensor, fidelity, mesolve
import numpy as np
import itertools


def compute_state_fidelity(
    qs0_list,
    target_gate, 
    Htotal, 
    collapse_list, 
    tlist,
    psi0_list=None,
    comp_indices=None,
    expect_list=None
):
    """
    Simulate gate dynamics for a list of initial states.
    
    Parameters:
    ----------
    qs0_list : list[Qobj]
        List of initial states in computational subspace (2-qubit, dim=2 each)
    target_gate : Qobj
        Target gate operator (e.g., CZ gate)
    Htotal : list or Qobj
        Time-dependent or time-independent Hamiltonian
    collapse_list : list
        Lindblad collapse operators
    tlist : array
        Time points for evolution
    psi0_list : list[Qobj], optional
        List of initial states in full Hilbert space (with auxiliary levels).
        If None, assumes psi0_list = qs0_list (no auxiliary levels)
        Must have same length as qs0_list if provided
    comp_indices : list, optional
        Indices of computational basis states in the full Hilbert space.
        Required if psi0_list is provided and different from qs0_list.
    expect_list : list, optional
        Expectation operators to track during evolution.
        If None, will track psi0 * psi0.dag() for each state
        
    Returns:
    -------
    fidelity_list : list[float]
        State fidelities between target and actual final states for each input
    result_list : list[QuTiP Result object]
        Full simulation results for each input state
    
    Examples:
    --------
    # Case 1: List of qs0 only (no auxiliary levels)
    >>> fid_list, result_list = compute_state_fidelity(
    ...     [qs0_1, qs0_2, qs0_3], target_gate, Htotal, collapse_list, tlist
    ... )
    
    # Case 2: With auxiliary levels
    >>> fid_list, result_list = compute_state_fidelity(
    ...     [qs0_1, qs0_2], target_gate, Htotal, collapse_list, tlist,
    ...     psi0_list=[psi0_full_1, psi0_full_2], comp_indices=[0, 1, 4, 5]
    ... )
    """
    # Validate inputs
    if psi0_list is not None and len(psi0_list) != len(qs0_list):
        raise ValueError("psi0_list must have same length as qs0_list")
    
    # Determine if we need projection
    use_projection = (psi0_list is not None and comp_indices is not None)
    
    fidelity_list = []
    result_list = []
    
    # Process each initial state
    for idx, qs0 in enumerate(qs0_list):
        # Determine the initial state for evolution
        if psi0_list is None:
            psi0 = qs0
        else:
            psi0 = psi0_list[idx]
        
        # Setup expectation operators
        if expect_list is None:
            expect_ops = [psi0 * psi0.dag()]
        else:
            expect_ops = expect_list.copy()
            expect_ops.append(psi0 * psi0.dag())
        
        # Get target final state
        target_qsf = target_gate * qs0
        target_dmf = target_qsf * target_qsf.dag()
        
        # Simulate dynamics
        result = mesolve(
            Htotal, psi0, tlist, collapse_list, 
            e_ops=expect_ops, 
            options={"store_final_state": True, "store_states": False}
        )
        
        # Get final state
        my_dmf = result.final_state
        
        # Project to computational subspace if needed
        if use_projection:
            my_dmf = my_dmf.full()
            my_dmf = my_dmf[np.ix_(comp_indices, comp_indices)]
            my_dmf = Qobj(my_dmf, dims=[qs0.dims[0],qs0.dims[0]])
        
        # Compute fidelity
        fid = fidelity(target_dmf, my_dmf)
        
        fidelity_list.append(float(fid))
        result_list.append(result)
    
    return fidelity_list, result_list


def compute_gate_fidelity_arithmetic(state_fidelity_list: list, dim: int):
    # calculate the average gate fidelity with arithmetic mean
    fidelity_arith = 1 / ( dim + 1 ) * ( sum(state_fidelity_list) )
    return fidelity_arith


def compute_gate_fidelity_geometric(state_fidelity_list: list, dim: int):
    # calculate the average gate fidelity with geometric mean
    fidelity_prod = np.prod(state_fidelity_list[:-1])
    fidelity_geom = 1 / ( dim + 1 ) + (1 - 1 / ( dim + 1 )) \
                  * state_fidelity_list[-1] * fidelity_prod
    return fidelity_geom


def compute_gate_fidelity_mixed(state_fidelity_list: list, dim: int, return_all: bool = False):
    """
    Calculate the mixed average gate fidelity.
    
    Parameters:
    ----------
    state_fidelity_list : list
        List of state fidelities
    dim : int
        Dimension of the computational space (e.g., 4 for 2-qubit gate)
    return_all : bool, optional
        If True, return (fidelity_mixed, fidelity_geom, fidelity_arith)
        If False, return only fidelity_mixed (default)
    
    Returns:
    -------
    fidelity_mixed : float
        Mixed fidelity (if return_all=False)
    (fidelity_mixed, fidelity_geom, fidelity_arith) : tuple
        All three fidelity measures (if return_all=True)
    """
    # calculate the average gate fidelity with a combined measure of two
    fidelity_prod = np.prod(state_fidelity_list[:-1])
    fidelity_geom = 1 / ( dim + 1 ) + (1 - 1 / ( dim + 1 )) \
                  * state_fidelity_list[-1] * fidelity_prod
    fidelity_arith = 1 / ( dim + 1 ) * ( sum(state_fidelity_list) )

    lambda_mix = 1 - (1 - fidelity_prod) / (1 - fidelity_prod * state_fidelity_list[-1])
    # lambda_mix = fidelity_prod * (1 - state_fidelity_list[-1]) / (1 - fidelity_prod * state_fidelity_list[-1])
    fidelity_mixed = lambda_mix * fidelity_geom + (1 - lambda_mix) * fidelity_arith

    if return_all:
        return fidelity_mixed, fidelity_geom, fidelity_arith
    else:
        return fidelity_mixed