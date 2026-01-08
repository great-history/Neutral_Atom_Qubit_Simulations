import time
import pickle
import json
from pathlib import Path
from functools import partial
import numpy as np
from hamiltonian_builder import construct_TD_TAHam, construct_TD_SAHam
from fidelity_calculator import compute_state_fidelity, \
    compute_gate_fidelity_mixed, compute_gate_fidelity_geometric, compute_gate_fidelity_arithmetic


# only the pulse parameters are what we want to optimize here
def TQ_fidelity_objective_core(
    pulse_params_array,
    pulse_type,
    pulse_functions,
    atom0_base_params,
    atom1_base_params,
    lindblad_params,
    coupling_params,
    target_gate,
    qs0_list,
    psi0_list=None,
    comp_indices=None,
    expect_list=None,
    fidelity_type='mixed',
    num_time_points=300,
    use_log=False
    ):
    """
    Core objective function for TWO-QUBIT gate pulse optimization.
    
    This function is specifically designed for two-qubit gates that require
    Rydberg blockade coupling between atoms.
    
    Parameters:
    -----------
    pulse_params_array : array-like
        Tunable pulse parameters. Format depends on pulse_type:
        - 'ARP': [T_gate, tau_ratio, amp_pulse1, amp_pulse2, ...]
          where tau = tau_ratio * T_gate, tau_ratio ∈ [0, 0.2]
        - 'square': [T_gate, amp_pulse1, amp_pulse2, ...]
        - 'gaussian': [T_gate, sigma, amp_pulse1, amp_pulse2, ...]
    coupling_params : dict
        Fixed coupling parameters, must include 'Rydberg_B' for two-qubit interaction
    pulse_type : str
        Type of pulse: 'ARP', 'square', 'gaussian', etc.
    pulse_functions : dict
        Dictionary containing pulse functions (keys are pulse names used in Hamiltonian)
        Example: {'Omega_r': func1, 'Delta_r': func2}
    atom0_base_params : dict
        Base parameters for atom 0 (Omega_01, delta_1, etc.)
    atom1_base_params : dict
        Base parameters for atom 1
    lindblad_params : dict
        Lindblad operator parameters
    target_gate : Qobj
        Target gate operator
    qs0_list : list
        Initial states in computational subspace
    psi0_list : list, optional
        Initial states in full Hilbert space
    comp_indices : list, optional
        Computational basis indices
    expect_list : list, optional
        Expectation operators
    fidelity_type : str, optional
        Type of fidelity to compute: 'mixed' (default), 'geom' (geometric), or 'arith' (arithmetic)
    num_time_points : int
        Number of time points for simulation
    use_log : bool, optional
        If True, return log(1-F) for better numerical stability at high fidelities (default: False)
    
    Returns:
    --------
    objective_value : float
        1 - fidelity (if use_log=False) or log(1-fidelity) (if use_log=True)
    
    Note:
    -----
    For single-qubit gate optimization, use SQ_fidelity_objective_core instead.
    """
    
    # Dimension of two-qubit computational space
    dim_qubits=4

    # Get pulse names from the dictionary
    pulse_names = list(pulse_functions.keys())
    num_pulses = len(pulse_names)
    
    # Parse pulse parameters based on type
    if pulse_type.upper() == 'ARP':
        if len(pulse_params_array) != 2 + num_pulses:
            raise ValueError(f"ARP pulse expects {2 + num_pulses} parameters "
                           f"(T_gate, tau_ratio, {num_pulses} amplitudes), "
                           f"got {len(pulse_params_array)}")
        
        T_gate = pulse_params_array[0]
        tau_ratio = pulse_params_array[1]
        tau = tau_ratio * T_gate  # Calculate tau from ratio
        amplitudes = pulse_params_array[2:]
        
        pulse_args_dict = {}
        for i, pulse_name in enumerate(pulse_names):
            pulse_args_dict[pulse_name] = {
                f'amp_{pulse_name}': amplitudes[i],
                'T_gate': T_gate,
                'tau': tau
            }
        
    elif pulse_type.upper() == 'SQUARE':
        if len(pulse_params_array) != 1 + num_pulses:
            raise ValueError(f"Square pulse expects {1 + num_pulses} parameters "
                           f"(T_gate, {num_pulses} amplitudes), "
                           f"got {len(pulse_params_array)}")
        
        T_gate = pulse_params_array[0]
        amplitudes = pulse_params_array[1:]
        
        pulse_args_dict = {}
        for i, pulse_name in enumerate(pulse_names):
            pulse_args_dict[pulse_name] = {
                f'amp_{pulse_name}': amplitudes[i],
                'T_gate': T_gate
            }
        
    elif pulse_type.upper() == 'GAUSSIAN':
        if len(pulse_params_array) != 2 + num_pulses:
            raise ValueError(f"Gaussian pulse expects {2 + num_pulses} parameters "
                           f"(T_gate, sigma, {num_pulses} amplitudes), "
                           f"got {len(pulse_params_array)}")
        
        T_gate = pulse_params_array[0]
        sigma = pulse_params_array[1]
        amplitudes = pulse_params_array[2:]
        
        pulse_args_dict = {}
        for i, pulse_name in enumerate(pulse_names):
            pulse_args_dict[pulse_name] = {
                f'amp_{pulse_name}': amplitudes[i],
                'T_gate': T_gate,
                'sigma': sigma
            }
        
    else:
        raise ValueError(f"Unknown pulse type: {pulse_type}")
    
    # Update atom parameters with all pulse information
    atom0_params = {**atom0_base_params}
    atom1_params = {**atom1_base_params}
    
    for pulse_name in pulse_names:
        atom0_params[pulse_name] = (pulse_functions[pulse_name], pulse_args_dict[pulse_name])
        atom1_params[pulse_name] = (pulse_functions[pulse_name], pulse_args_dict[pulse_name])
    
    # Create time list
    tlist = np.linspace(0, T_gate, num_time_points + 1)
    
    try:
        
        # Construct TWO-ATOM Hamiltonian with Rydberg coupling
        Htotal, collapse_list = construct_TD_TAHam(
            atom0_params, atom1_params,
            lindblad_params, lindblad_params,
            coupling_params['Rydberg_B']  # Two-qubit coupling
        )
        
        state_fidelity_list, _ = compute_state_fidelity(
            qs0_list, target_gate, Htotal, collapse_list, tlist,
            psi0_list=psi0_list, comp_indices=comp_indices, expect_list=expect_list
        )
        
        # 根据 fidelity_type 选择不同的保真度
        fidelity_type_lower = fidelity_type.lower()  # 转换为小写
        
        if fidelity_type_lower == 'mixed':
            fidelity = compute_gate_fidelity_mixed(
                state_fidelity_list, dim_qubits, return_all=False
            )
        elif fidelity_type_lower == 'geom':
            fidelity = compute_gate_fidelity_geometric(
                state_fidelity_list, dim_qubits
            )
        elif fidelity_type_lower == 'arith':
            fidelity = compute_gate_fidelity_arithmetic(
                state_fidelity_list, dim_qubits
            )
        else:
            raise ValueError(f"Unknown fidelity_type: {fidelity_type}")
        
        infidelity = np.abs(1.0 - fidelity)
        if use_log:
            return np.log(infidelity + 1e-15)  # Add epsilon to avoid log(0)
        else:
            return infidelity
        
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 0.0 if use_log else 1.0


def create_TQ_pulse_optimizer(
    pulse_type,
    pulse_functions,
    atom0_base_params,
    atom1_base_params,
    lindblad_params,
    coupling_params,
    target_gate,
    qs0_list,
    psi0_list=None,
    comp_indices=None,
    expect_list=None,
    fidelity_type='mixed',
    num_time_points=300,
    use_log=False
):
    """
    Create a TWO-QUBIT gate pulse optimization objective function.
    
    Parameters:
    -----------
    pulse_type : str
        Type of pulse: 'ARP', 'square', 'gaussian', etc.
    pulse_functions : dict
        Dictionary containing pulse functions (keys are pulse names)
        Example: {'Omega_r': APR_pulse_Omega_r, 'Delta_r': APR_pulse_Delta_r}
    target_gate : Qobj
        Target gate operator
    atom0_base_params : dict
        Base parameters for atom 0 (Omega_01, delta_1, etc.)
    atom1_base_params : dict
        Base parameters for atom 1
    lindblad_params : dict
        Lindblad operator parameters
    coupling_params : dict
        Fixed coupling parameters, must include 'Rydberg_B'
    qs0_list : list
        Initial states in computational subspace
    psi0_list : list, optional
        Initial states in full Hilbert space
    comp_indices : list, optional
        Computational basis indices
    expect_list : list, optional
        Expectation operators
    fidelity_type : str, optional
        Type of fidelity to optimize: 'mixed' (default), 'geom', or 'arith'
    num_time_points : int
        Number of time points for simulation
    use_log : bool, optional
        If True, optimize log(1-F) for better stability at high fidelities (default: False)
    
    Returns:
    --------
    objective_func : callable
        Objective function that takes pulse_params_array and returns infidelity
    
    Examples:
    ---------
    >>> # Example 1: Optimize CZ gate with ARP pulse using mixed fidelity
    >>> from pulse_functions import APR_pulse_Omega_r, APR_pulse_Delta_r
    >>> 
    >>> pulse_funcs = {
    ...     'Omega_r': APR_pulse_Omega_r,
    ...     'Delta_r': APR_pulse_Delta_r
    ... }
    >>> 
    >>> objective = create_TQ_pulse_optimizer(
    ...     pulse_type='ARP',
    ...     pulse_functions=pulse_funcs,
    ...     target_gate=CZ_gate,
    ...     atom0_base_params={'Omega_01': 0, 'delta_1': 0},
    ...     atom1_base_params={'Omega_01': 0, 'delta_1': 0},
    ...     lindblad_params={'gamma_r': 1/540, 'b_0r': 1/16, 'b_1r': 1/16, 'b_dr': 7/8},
    ...     coupling_params={'Rydberg_B': 200*2*np.pi},
    ...     qs0_list=qs0_list,
    ...     psi0_list=psi0_list,
    ...     comp_indices=[0, 1, 4, 5],
    ...     fidelity_type='mixed'
    ... )
    """
    return partial(
        TQ_fidelity_objective_core,
        pulse_type=pulse_type,
        pulse_functions=pulse_functions,
        target_gate=target_gate,
        atom0_base_params=atom0_base_params,
        atom1_base_params=atom1_base_params,
        lindblad_params=lindblad_params,
        coupling_params=coupling_params,
        qs0_list=qs0_list,
        psi0_list=psi0_list,
        comp_indices=comp_indices,
        expect_list=expect_list,
        fidelity_type=fidelity_type,  # 传递 fidelity_type 参数
        num_time_points=num_time_points,
        use_log=use_log
    )


def SQ_fidelity_objective_core(
    pulse_params_array,
    pulse_type,
    pulse_functions,
    atom_base_params,
    lindblad_params,
    target_gate,
    qs0_list,
    psi0_list=None,
    comp_indices=None,
    expect_list=None,
    fidelity_type='mixed',
    num_time_points=300,
    use_log=False
):
    """
    Core objective function for SINGLE-QUBIT gate pulse optimization.
    
    This function is designed for single-qubit gates without inter-atom coupling.
    
    Parameters:
    -----------
    pulse_params_array : array-like
        Tunable pulse parameters. Format depends on pulse_type:
        - 'Gaussian': [T_gate, sigma, amp_pulse1, amp_pulse2, ...]
        - 'Window': [T_gate, amp_pulse1, amp_pulse2, ...]
    pulse_type : str
        Type of pulse: 'Gaussian' or 'Window' (case-insensitive)
    pulse_functions : dict
        Dictionary containing pulse functions (keys are pulse names used in Hamiltonian)
        Example: {'Omega_r': gaussian_pulse_func, 'Delta_r': gaussian_pulse_func}
    atom_base_params : dict
        Base parameters for the atom (Omega_01, delta_1, etc.)
    lindblad_params : dict
        Lindblad operator parameters (gamma_r, b_0r, b_1r, b_dr)
    target_gate : Qobj
        Target single-qubit gate operator (e.g., X_gate, Y_gate, Z_gate)
    qs0_list : list
        Initial states in computational subspace
    psi0_list : list, optional
        Initial states in full Hilbert space (with auxiliary levels)
    comp_indices : list, optional
        Computational basis indices for projection
    expect_list : list, optional
        Expectation operators
    fidelity_type : str, optional
        Type of fidelity to compute: 'mixed' (default), 'geom', or 'arith'
    num_time_points : int
        Number of time points for simulation
    use_log : bool, optional
        If True, return log(1-F) for better numerical stability at high fidelities (default: False)
    
    Returns:
    --------
    objective_value : float
        1 - fidelity (if use_log=False) or log(1-fidelity) (if use_log=True)
    
    Note:
    -----
    For two-qubit gate optimization, use TQ_fidelity_objective_core instead.
    
    Examples:
    ---------
    >>> # Example: Optimize X gate with Gaussian pulse
    >>> from pulse_functions import gaussian_pulse
    >>> pulse_funcs = {'Omega_r': gaussian_pulse, 'Delta_r': gaussian_pulse}
    >>> 
    >>> infidelity = SQ_fidelity_objective_core(
    ...     pulse_params_array=[0.5, 0.1, 20*2*np.pi, 0],  # [T_gate, sigma, amp_Omega_r, amp_Delta_r]
    ...     pulse_type='Gaussian',
    ...     pulse_functions=pulse_funcs,
    ...     atom_base_params={'Omega_01': 0, 'delta_1': 0},
    ...     lindblad_params={'gamma_r': 1/540, 'b_0r': 1/16, 'b_1r': 1/16, 'b_dr': 7/8},
    ...     target_gate=X_gate,
    ...     qs0_list=qs0_list,
    ...     psi0_list=psi0_list,
    ...     comp_indices=[0, 1],
    ...     fidelity_type='mixed'
    ... )
    """
    
    # Dimension of single-qubit computational space
    dim_qubits = 2

    # Get pulse names from the dictionary
    pulse_names = list(pulse_functions.keys())
    num_pulses = len(pulse_names)
    
    # Parse pulse parameters based on type
    if pulse_type.upper() == 'GAUSSIAN':
        if len(pulse_params_array) != 3:
            raise ValueError(f"Gaussian pulse expects 3 parameters "
                           f"(pulse_width, sigma, Omega_01), "
                           f"got {len(pulse_params_array)}")
        
        sigma = pulse_params_array[0]
        t0 = pulse_params_array[1] * sigma
        Omega_01 = pulse_params_array[2]
        # Create time list based on sigma
        tlist = np.linspace(0, sigma * 30, num_time_points + 1)
        
        # Assume single pulse (e.g., only Omega_r)
        pulse_name = pulse_names[0]
        pulse_args_dict = {
            pulse_name: {
                f'amp_{pulse_name}': Omega_01,
                'sigma': sigma,
                't0': t0
            }
        }
        
        print(f"")
    elif pulse_type.upper() == 'WINDOW':
        if len(pulse_params_array) != 2:
            raise ValueError(f"Window pulse expects 2 parameters "
                           f"(T_detuning, amp_delta_1), "
                           f"got {len(pulse_params_array)}")
        
        T_detuning = pulse_params_array[0]
        amp_delta_1 = pulse_params_array[1]
        
        # Assume single pulse (e.g., only delta_1)
        pulse_name = pulse_names[0]
        pulse_args_dict = {
            pulse_name: {
                'amp_delta_1': amp_delta_1,
                'T_detuning': T_detuning
            }
        }

        # Create time list based on T_detuning
        tlist = np.linspace(0, 1.1, num_time_points + 1) * T_detuning
        
    else:
        raise ValueError(f"Unknown pulse type: {pulse_type}. "
                       f"Expected 'Gaussian' or 'Window'")
    
    # Update atom parameters with all pulse information
    atom_params = {**atom_base_params}
    
    for pulse_name in pulse_names:
        atom_params[pulse_name] = (pulse_functions[pulse_name], pulse_args_dict[pulse_name])
    # print(f"Atom parameters with pulse info: {atom_params}")
    try:
        # Construct SINGLE-ATOM Hamiltonian (no coupling)
        Htotal, collapse_list = construct_TD_SAHam(
            atom_params, lindblad_params
        )
        
        # print(f"Constructed Hamiltonian and collapse operators successfully.")
        state_fidelity_list, _ = compute_state_fidelity(
            qs0_list, target_gate, Htotal, collapse_list, tlist,
            psi0_list=psi0_list, comp_indices=comp_indices, expect_list=expect_list
        )
        
        # Choose fidelity type
        fidelity_type_lower = fidelity_type.lower()
        
        if fidelity_type_lower == 'mixed':
            fidelity = compute_gate_fidelity_mixed(
                state_fidelity_list, dim_qubits, return_all=False
            )
        elif fidelity_type_lower == 'geom':
            fidelity = compute_gate_fidelity_geometric(
                state_fidelity_list, dim_qubits
            )
        elif fidelity_type_lower == 'arith':
            fidelity = compute_gate_fidelity_arithmetic(
                state_fidelity_list, dim_qubits
            )
        else:
            raise ValueError(f"Unknown fidelity_type: {fidelity_type}")
        
        infidelity = 1.0 - fidelity
        if use_log:
            return np.log(infidelity + 1e-15)  # Add epsilon to avoid log(0)
        else:
            return infidelity
        
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 0.0 if use_log else 1.0


def create_SQ_pulse_optimizer(
    pulse_type,
    pulse_functions,
    atom_base_params,
    lindblad_params,
    target_gate,
    qs0_list,
    psi0_list=None,
    comp_indices=None,
    expect_list=None,
    fidelity_type='mixed',
    num_time_points=300,
    use_log=False
):
    """
    Create a SINGLE-QUBIT gate pulse optimization objective function.
    
    Parameters:
    -----------
    pulse_type : str
        Type of pulse: 'Gaussian' or 'Window'
    pulse_functions : dict
        Dictionary containing pulse functions (keys are pulse names)
        Example: {'Omega_r': gaussian_pulse, 'Delta_r': gaussian_pulse}
    atom_base_params : dict
        Base parameters for the atom (Omega_01, delta_1, etc.)
    lindblad_params : dict
        Lindblad operator parameters (gamma_r, b_0r, b_1r, b_dr)
    target_gate : Qobj
        Target single-qubit gate operator (e.g., X_gate)
    qs0_list : list
        Initial states in computational subspace
    psi0_list : list, optional
        Initial states in full Hilbert space
    comp_indices : list, optional
        Computational basis indices
    expect_list : list, optional
        Expectation operators
    fidelity_type : str, optional
        Type of fidelity to optimize: 'mixed' (default), 'geom', or 'arith'
    num_time_points : int
        Number of time points for simulation
    use_log : bool, optional
        If True, optimize log(1-F) for better stability at high fidelities (default: False)
    
    Returns:
    --------
    objective_func : callable
        Objective function that takes pulse_params_array and returns infidelity
    
    Examples:
    ---------
    >>> # Example 1: Optimize X gate with Gaussian pulse
    >>> from pulse_functions import gaussian_pulse
    >>> 
    >>> pulse_funcs = {'Omega_r': gaussian_pulse, 'Delta_r': gaussian_pulse}
    >>> 
    >>> objective = create_SQ_pulse_optimizer(
    ...     pulse_type='Gaussian',
    ...     pulse_functions=pulse_funcs,
    ...     atom_base_params={'Omega_01': 0, 'delta_1': 0},
    ...     lindblad_params={'gamma_r': 1/540, 'b_0r': 1/16, 'b_1r': 1/16, 'b_dr': 7/8},
    ...     target_gate=X_gate,
    ...     qs0_list=qs0_list,
    ...     psi0_list=psi0_list,
    ...     comp_indices=[0, 1],
    ...     fidelity_type='mixed',
    ...     num_time_points=300
    ... )
    >>> 
    >>> # Optimize
    >>> from scipy.optimize import minimize
    >>> x0 = [0.5, 0.1, 20*2*np.pi, 0]  # [T_gate, sigma, amp_Omega_r, amp_Delta_r]
    >>> result = minimize(objective, x0, method='Nelder-Mead')
    
    >>> # Example 2: Optimize with monitoring
    >>> param_names = ['T_gate', 'sigma', 'amp_Omega_r', 'amp_Delta_r']
    >>> monitor = OptimizationMonitor(param_names, objective)
    >>> result = minimize(objective, x0, method='Nelder-Mead', callback=monitor)
    >>> monitor.print_summary()
    """
    return partial(
        SQ_fidelity_objective_core,
        pulse_type=pulse_type,
        pulse_functions=pulse_functions,
        atom_base_params=atom_base_params,
        lindblad_params=lindblad_params,
        target_gate=target_gate,
        qs0_list=qs0_list,
        psi0_list=psi0_list,
        comp_indices=comp_indices,
        expect_list=expect_list,
        fidelity_type=fidelity_type,
        num_time_points=num_time_points,
        use_log=use_log
    )


class OptimizationMonitor:
    """Monitor and record pulse optimization progress"""
    
    def __init__(self, param_names, objective_func, verbose=True):
        """
        Parameters:
        -----------
        param_names : list
            Names of pulse parameters being optimized
        objective_func : callable
            Objective function to evaluate
        verbose : bool, optional
            If True, print progress at each iteration. Default is True.
        """
        self.param_names = param_names
        self.objective_func = objective_func
        self.iteration = 0
        self.verbose = verbose
        self.history = {
            'iteration': [],
            'fidelity': [],
            'infidelity': [],
            **{name: [] for name in param_names}
        }
        
        # 计时器
        self.start_time = None
        self.end_time = None
        self.total_time = None
    
    def __call__(self, xk):
        """Callback function called at each optimization iteration"""
        # 第一次调用时开始计时
        if self.iteration == 0:
            self.start_time = time.time()
        
        self.iteration += 1
        
        # Compute fidelity
        infidelity = self.objective_func(xk)
        fidelity = 1.0 - infidelity
        
        # Store results
        self.history['iteration'].append(self.iteration)
        self.history['fidelity'].append(fidelity)
        self.history['infidelity'].append(infidelity)
        
        for i, name in enumerate(self.param_names):
            self.history[name].append(xk[i])
        
        # Print progress (only if verbose=True)
        if self.verbose:
            params_str = ", ".join([f"{name}={xk[i]:.4f}" for i, name in enumerate(self.param_names)])
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"Iter {self.iteration}: F={fidelity:.6f} | {params_str} | Time: {elapsed:.1f}s")
    
    def finalize(self):
        """Finalize timing (called after optimization completes)"""
        if self.start_time is not None and self.end_time is None:
            self.end_time = time.time()
            self.total_time = self.end_time - self.start_time
    
    def get_best_result(self):
        """Get the best result from optimization history"""
        best_idx = np.argmax(self.history['fidelity'])
        best_params = {name: self.history[name][best_idx] for name in self.param_names}
        return {
            'iteration': self.history['iteration'][best_idx],
            'fidelity': self.history['fidelity'][best_idx],
            'infidelity': self.history['infidelity'][best_idx],
            'params': best_params
        }
    
    def print_summary(self):
        """Print optimization summary"""
        # 确保计时结束
        self.finalize()
        
        best = self.get_best_result()
        
        print("\n" + "="*70)
        print("Optimization Summary")
        print("="*70)
        print(f"Total iterations: {self.iteration}")
        print(f"Best fidelity: {best['fidelity']:.8f} (at iteration {best['iteration']})")
        
        # 打印用时
        if self.total_time is not None:
            hours, remainder = divmod(self.total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            if hours > 0:
                time_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
            elif minutes > 0:
                time_str = f"{int(minutes)}m {seconds:.1f}s"
            else:
                time_str = f"{seconds:.1f}s"
            
            print(f"Total time: {time_str} ({self.total_time:.2f} seconds)")
            print(f"Time per iteration: {self.total_time/self.iteration:.2f} seconds")
        
        print(f"\nBest parameters:")
        for name, value in best['params'].items():
            print(f"  {name:15s} = {value:12.6f}")
        print("="*70 + "\n")
    
    def save(self, filepath, format='pickle'):
        """
        Save optimization monitor data to file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to save file
        format : str, optional
            Save format: 'pickle' (default) or 'json'
            - 'pickle': Saves entire object (can restore with objective_func)
            - 'json': Saves only history data (human-readable)
        
        Examples:
        ---------
        >>> monitor.save('optimization_results.pkl')
        >>> monitor.save('optimization_results.json', format='json')
        """
        filepath = Path(filepath)
        
        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 完成计时
        self.finalize()
        
        if format.lower() == 'pickle':
            # 保存整个对象（包括 objective_func）
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"✅ Saved optimization monitor to: {filepath}")
            
        elif format.lower() == 'json':
            # 只保存数据（不包括 objective_func）
            data = {
                'param_names': self.param_names,
                'iteration': self.iteration,
                'total_time': self.total_time,
                'history': {k: [float(x) for x in v] for k, v in self.history.items()},
                'best_result': self.get_best_result()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Saved optimization data to: {filepath}")
        
        else:
            raise ValueError(f"Unknown format: {format}. Use 'pickle' or 'json'")
    
    @classmethod
    def load(cls, filepath, objective_func=None):
        """
        Load optimization monitor from file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to load file
        objective_func : callable, optional
            Objective function (required if loading from JSON)
        
        Returns:
        --------
        monitor : OptimizationMonitor
            Loaded optimization monitor
        
        Examples:
        ---------
        >>> # Load from pickle (objective_func included)
        >>> monitor = OptimizationMonitor.load('optimization_results.pkl')
        >>> 
        >>> # Load from JSON (need to provide objective_func)
        >>> monitor = OptimizationMonitor.load('optimization_results.json', 
        ...                                      objective_func=my_objective)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # 根据文件扩展名判断格式
        if filepath.suffix in ['.pkl', '.pickle']:
            # 从 pickle 加载完整对象
            with open(filepath, 'rb') as f:
                monitor = pickle.load(f)
            print(f"✅ Loaded optimization monitor from: {filepath}")
            print(f"   Total iterations: {monitor.iteration}")
            print(f"   Best fidelity: {monitor.get_best_result()['fidelity']:.8f}")
            return monitor
            
        elif filepath.suffix == '.json':
            # 从 JSON 加载数据
            if objective_func is None:
                raise ValueError("objective_func is required when loading from JSON")
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 重建 monitor 对象
            monitor = cls(data['param_names'], objective_func, verbose=False)
            monitor.iteration = data['iteration']
            monitor.total_time = data['total_time']
            monitor.history = {k: list(v) for k, v in data['history'].items()}
            
            print(f"✅ Loaded optimization data from: {filepath}")
            print(f"   Total iterations: {monitor.iteration}")
            print(f"   Best fidelity: {data['best_result']['fidelity']:.8f}")
            return monitor
        
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")
    
    def plot_convergence(self, figsize=(12, 8), save_path=None):
        """
        Plot optimization convergence.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height). Default is (12, 8)
        save_path : str or Path, optional
            Path to save the figure. If None, figure is not saved.
            Supported formats: .png, .pdf, .svg, .jpg, etc.
        
        Returns:
        --------
        fig, axes : matplotlib figure and axes objects
        
        Examples:
        ---------
        >>> # Plot without saving
        >>> fig, axes = monitor.plot_convergence()
        >>> 
        >>> # Plot and save
        >>> fig, axes = monitor.plot_convergence(save_path='results/convergence.png')
        """
        from plotting_helpers import plot_optimization_convergence
        fig, axes = plot_optimization_convergence(self.history, self.param_names, figsize)
        
        # Save figure if path is provided
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved convergence plot to: {save_path}")
        
        return fig, axes
    
    def get_best_result_summary(self):
        """
        Get the best result summary from optimization history.
        
        Returns:
            dict: Dictionary containing details of the best iteration
        """
        if not self.history['fidelity']:
            return None
        
        best_iter = np.argmax(self.history['fidelity'])
        
        summary = {
            'total_iterations': len(self.history['iteration']),
            'best_iteration': self.history['iteration'][best_iter],
            'best_fidelity': self.history['fidelity'][best_iter],
            'best_infidelity': self.history['infidelity'][best_iter],
            'best_parameters': {}
        }
        
        # Extract best parameters
        for name in self.param_names:
            summary['best_parameters'][name] = self.history[name][best_iter]
        
        return summary
    
    def print_best_result(self, convert_amp_to_MHz=True):
        """
        Print formatted summary of the best result.
        
        Args:
            convert_amp_to_MHz: Whether to convert amplitude parameters to MHz
        """
        summary = self.get_best_result_summary()
        
        if summary is None:
            print("❌ No optimization history available")
            return
        
        print(f"📊 Optimization Summary:")
        print(f"  Total iterations: {summary['total_iterations']}")
        print(f"  Best iteration: {summary['best_iteration']}")
        print(f"  Best fidelity: {summary['best_fidelity']:.6f}")
        print(f"  Best infidelity: {summary['best_infidelity']:.2e}")
        print(f"\n  Best parameters:")
        
        for name, value in summary['best_parameters'].items():
            if convert_amp_to_MHz and 'amp' in name:
                print(f"    {name}: {value/(2*np.pi):.4f} MHz")
            else:
                print(f"    {name}: {value:.4f}")