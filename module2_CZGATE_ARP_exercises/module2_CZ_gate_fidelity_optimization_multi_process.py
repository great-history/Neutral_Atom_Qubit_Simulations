"""
Module 2: Finding Neutral Atom Qubit Gates
CZ-gate fidelity optimization with multi-process parallelization

This script performs parallel optimization of CZ gate fidelity across different
Rydberg blockade strengths (B values) using joblib for efficient multiprocessing.

Code Structure:
- setup_parameters(): Initialize physical parameters for the quantum system
- optimize_for_single_B(): Optimization function for a single B value (parallelized)
- main(): Main execution flow with parallel optimization
- if __name__ == '__main__': Entry point with error handling

This modular structure is essential for multiprocessing compatibility on Windows,
where the spawn method requires functions to be importable without side effects.

Note:
- Jupyter notebooks may have issues with multiprocessing on Windows. 
  It is recommended to run this script directly via Python interpreter.
- The main() function encapsulation prevents infinite recursion when child 
  processes import the module during parallel execution.

Usage:
    python module2_CZ_gate_fidelity_optimization_multi_process.py
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from joblib import Parallel, delayed
import sys
from multiprocessing import cpu_count
from datetime import datetime
import warnings
from scipy.linalg import LinAlgWarning

# # Filter out expected numerical warnings from QuTiP
# warnings.filterwarnings('ignore', category=LinAlgWarning, 
#                        message='.*Matrix is singular.*')

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
# Import custom modules
from common_imports import *
# from ...pulse_functions import *
from myPkg.optimization_utils import *
from myPkg.utils import *


def optimize_for_single_B_single_init(task_idx, init_idx, scale_B, Rydberg_B, initial_params, bounds, save_dir_str,
                                      atom0_ham_params, atom1_ham_params, lindblad_params,
                                      target_gate, qs0_list, psi0_list, comp_indices, total_tasks):
    """
    Optimize gate fidelity for a single Rydberg blockade strength with a single initial parameter set.
    
    Parameters
    ----------
    task_idx : int
        Global task index (from 0 to total_tasks-1)
    init_idx : int
        Index of the initial parameter set (for multi-start optimization)
    scale_B : float
        Rydberg blockade strength [MHz] (for display)
    Rydberg_B : float
        Rydberg blockade strength [MHz * 2π]
    initial_params : dict
        Initial optimization parameters
    bounds : dict
        Parameter bounds for optimization
    save_dir_str : str
        Directory path to save results (as string for multiprocessing compatibility)
    atom0_ham_params : dict
        Hamiltonian parameters for atom 0
    atom1_ham_params : dict
        Hamiltonian parameters for atom 1
    lindblad_params : dict
        Lindblad dissipation parameters
    target_gate : Qobj
        Target gate operator
    qs0_list : list
        Initial states for fidelity calculation (qubit subspace)
    psi0_list : list
        Initial states for fidelity calculation (full space)
    comp_indices : list
        Indices of computational basis states
    total_tasks : int
        Total number of optimization tasks (n_B_values × n_init_sets)
        
    Returns
    -------
    dict
        Optimization results including fidelity and optimal parameters
    """
    # Convert string path back to Path object (important for Windows multiprocessing)
    save_dir = Path(save_dir_str)
    
    # Create subdirectory for this B value
    B_dir = save_dir / f'B{int(scale_B)}MHz'
    B_dir.mkdir(parents=True, exist_ok=True)
    
    # Save path for this specific initial parameter set
    save_path = B_dir / f'monitor_init{init_idx}.pkl'
    
    # Check if results already exist (optional: enable caching)
    # Uncomment to enable caching:
    # if save_path.exists():
    #     print(f"📂 [{task_idx+1}/{total_tasks}] Loading existing B = {scale_B:.0f} MHz (init {init_idx})")
    #     monitor = OptimizationMonitor.load(save_path)
    #     result = monitor.get_best_result()
    #     result['Rydberg_B_MHz'] = scale_B
    #     result['init_idx'] = init_idx
    #     return result
    
    print(f"🚀 [{task_idx+1}/{total_tasks}] Starting B = {scale_B:.0f} MHz, init={init_idx} (PID: {os.getpid()})")
    start_time = time.time()
    
    # Create objective function
    coupling_params = {'Rydberg_B': Rydberg_B}
    objective_func = create_TQ_pulse_optimizer(
        pulse_type='ARP',
        pulse_functions={'Omega_r': APR_pulse_Omega_r, 'Delta_r': APR_pulse_Delta_r},
        atom0_base_params=atom0_ham_params,
        atom1_base_params=atom1_ham_params,
        lindblad_params=lindblad_params,
        coupling_params=coupling_params,
        target_gate=target_gate,
        qs0_list=qs0_list,
        psi0_list=psi0_list,
        comp_indices=comp_indices,
        expect_list=None,
        fidelity_type='mixed',
        num_time_points=500,
        use_log=False
    )
    
    # Optimization parameters
    param_names = ['T_gate', 'tau_ratio', 'amp_Omega_r', 'amp_Delta_r']
    x0 = np.array([initial_params[name] for name in param_names])
    param_bounds = [bounds[name] for name in param_names]

    # options = {'maxiter': 300, 'disp': False, 'fatol': 1e-4, 'xatol': 1e-3}
    # Add initial_simplex scaling to improve numerical stability
    options = {'maxiter': 300, 'disp': False, 'fatol': 1e-4, 'xatol': 1e-3, 
               'adaptive': True}  # Adaptive parameters for better convergence
    
    # Run optimization
    monitor = OptimizationMonitor(param_names, objective_func, verbose=False)
    try:
        result = minimize(objective_func, x0, method='Nelder-Mead',
                          bounds=param_bounds, options=options, callback=monitor)
    except Exception as e:
        print(f"❌ [{task_idx+1}/{total_tasks}] Error for B = {scale_B:.0f} MHz, init={init_idx}: {str(e)}")
        return None
    
    # Get best result
    best = monitor.get_best_result()
    best['Rydberg_B_MHz'] = scale_B
    best['init_idx'] = init_idx
    
    # Save results
    try:
        # Save this specific optimization run
        monitor.save(save_path)
    except Exception as e:
        print(f"⚠️  [{task_idx+1}/{total_tasks}] Failed to save monitor for init {init_idx}: {str(e)}")
    
    elapsed = time.time() - start_time
    print(f"✅ [{task_idx+1}/{total_tasks}] Completed B = {scale_B:.0f} MHz, init={init_idx}, "
          f"F = {best['fidelity']:.6f}, Time = {elapsed:.1f}s")
    
    return best


def setup_parameters():
    """Set up all parameters for the optimization."""
    
    # Pulse parameters
    amp_Omega_r_list = [8.5*2*np.pi, 17*2*np.pi, 34*2*np.pi]  # [MHz]
    Omega_r_pulse_args = dict(
        amp_Omega_r=np.nan, 
        T_gate=np.nan, 
        tau=np.nan
    )
    
    amp_Delta_r_list = [11.5*2*np.pi, 23*2*np.pi, 46*2*np.pi]  # [MHz]
    Delta_r_pulse_args = dict(
        amp_Delta_r=np.nan, 
        T_gate=np.nan, 
        tau=np.nan
    )
    
    # Hamiltonian parameters
    atom0_ham_params = dict(
        Omega_01=0, 
        delta_1=0, 
        Omega_r=(APR_pulse_Omega_r, Omega_r_pulse_args),
        Delta_r=(APR_pulse_Delta_r, Delta_r_pulse_args)
    )
    
    atom1_ham_params = dict(
        Omega_01=0, 
        delta_1=0, 
        Omega_r=(APR_pulse_Omega_r, Omega_r_pulse_args),
        Delta_r=(APR_pulse_Delta_r, Delta_r_pulse_args)
    )
    
    # Lindblad dissipation parameters
    lindblad_params = dict(
        gamma_r=1/540, 
        b_0r=1/16, 
        b_1r=1/16, 
        b_dr=7/8
    )
    
    # Define target gate (CZ gate)
    target_gate = np.diag([1, -1, -1, -1])
    target_gate = Qobj(target_gate, dims=[[2, 2], [2, 2]])
    
    # Initial state lists
    psi0_list = make_initial_list_for_gate_fidelity(num_qubits=2, dim_atom=4)
    qs0_list = make_initial_list_for_gate_fidelity(num_qubits=2, dim_atom=2)
    comp_indices = [0, 1, 4, 5]
    
    return (atom0_ham_params, atom1_ham_params, lindblad_params, 
            target_gate, qs0_list, psi0_list, comp_indices)


def main(scale_B_list, initial_params, bounds, task_name='CZ_gate_optimization_multi_process'):
    """
    Main function to run parallel optimization.
    
    Parameters
    ----------
    scale_B_list : list
        List of Rydberg blockade strengths to optimize [MHz].
        Example: [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    initial_params : dict or list of dict
        Initial optimization parameters. Can be either:
        - A single dict with keys: 'T_gate', 'tau_ratio', 'amp_Omega_r', 'amp_Delta_r'
        - A list of dicts, where each dict has the same keys.
        When a list is provided, optimization will be performed with each initial
        parameter set, and the best result (highest fidelity) will be selected.
        This helps avoid local minima by exploring different starting points.
    bounds : dict
        Parameter bounds for optimization, same keys as initial_params.
        Values are tuples (min, max).
    task_name : str, optional
        Name of the optimization task. Used to create a unique results folder.
        Default: 'CZ_gate_optimization_multi_process'
    
    Returns
    -------
    results : list
        List of optimization results for each B value. Each result is the best
        among all initial parameter sets tried.
    """
    
    # ========== Configuration ==========
    Rydberg_B_list = [scale_B * 2 * np.pi for scale_B in scale_B_list]
    
    # Convert initial_params to list format if it's a single dict
    if isinstance(initial_params, dict):
        initial_params_list = [initial_params]
        n_init_sets = 1
    else:
        initial_params_list = initial_params
        n_init_sets = len(initial_params_list)
    
    # ========== Optimized Parallel Configuration ==========
    # Dynamically determine optimal number of parallel jobs
    n_cores = cpu_count()
    n_B_values = len(Rydberg_B_list)
    n_tasks = n_B_values * n_init_sets  # Total number of optimization tasks
    
    # Intelligent n_jobs setting:
    # - If tasks < cores: use number of tasks (no wasted processes)
    # - Otherwise: use cores - 2 (leave resources for system)
    if n_tasks < n_cores:
        n_jobs = n_tasks
    else:
        n_jobs = max(1, n_cores - 2)
    
    # Pre-dispatch: avoid excessive memory usage
    pre_dispatch = min(2 * n_jobs, n_tasks)
    
    # ========== Setup ==========
    # Create save directory with timestamp and task name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{timestamp}_{task_name}"
    # Use absolute path to avoid working directory issues in multiprocessing
    save_dir = Path(__file__).parent / 'save_data' / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure directory is fully created (important for Windows multiprocessing)
    # Force filesystem sync by checking directory existence
    assert save_dir.exists(), f"Failed to create directory: {save_dir}"
    
    print(f"\n📁 Results will be saved to: {save_dir.absolute()}")
    
    # Setup physical parameters
    (atom0_ham_params, atom1_ham_params, lindblad_params,
     target_gate, qs0_list, psi0_list, comp_indices) = setup_parameters()
    
    # ========== Start Parallel Optimization ==========
    print(f"\n{'='*70}")
    print(f"🚀 Starting Parallel Optimization with joblib")
    print(f"{'='*70}")
    print(f"  Task name: {task_name}")
    print(f"  CPU cores available: {n_cores}")
    print(f"  Total B values: {n_B_values}")
    print(f"  B range: {scale_B_list[0]:.0f} - {scale_B_list[-1]:.0f} MHz")
    print(f"  Initial parameter sets: {n_init_sets}")
    print(f"  Total optimizations: {n_tasks}")
    print(f"  Parallel jobs: {n_jobs} (optimized)")
    print(f"  Pre-dispatch: {pre_dispatch}")
    print(f"  Backend: loky (optimized for Windows)")
    print(f"  Max iterations per optimization: 300")
    print(f"{'='*70}\n")
    
    total_start = time.time()
    
    # Create all optimization tasks (B value × initial params combinations) with global task index
    optimization_tasks = []
    task_idx = 0
    for idx, (scale_B, B) in enumerate(zip(scale_B_list, Rydberg_B_list)):
        for init_idx, init_params in enumerate(initial_params_list):
            optimization_tasks.append((task_idx, init_idx, scale_B, B, init_params))
            task_idx += 1
    
    # Run parallel optimization using joblib
    # verbose=10 shows progress bar, pre_dispatch controls task distribution
    all_results = Parallel(
        n_jobs=n_jobs,
        backend='loky',
        verbose=10,
        pre_dispatch=pre_dispatch,
        timeout=None  # No timeout for long optimization tasks
    )(
        delayed(optimize_for_single_B_single_init)(
            task_idx, init_idx, scale_B, B, init_params, bounds, str(save_dir),
            atom0_ham_params, atom1_ham_params, lindblad_params,
            target_gate, qs0_list, psi0_list, comp_indices, n_tasks
        )
        for task_idx, init_idx, scale_B, B, init_params in optimization_tasks
    )
    
    total_time = time.time() - total_start
    
    # Filter out None results (failed optimizations)
    all_results = [r for r in all_results if r is not None]
    
    # Group results by B value and select the best one for each B
    results_by_B = {}
    for result in all_results:
        B_val = result['Rydberg_B_MHz']
        if B_val not in results_by_B:
            results_by_B[B_val] = []
        results_by_B[B_val].append(result)
    
    # Select best result (highest fidelity) for each B value
    results = []
    for B_val in sorted(results_by_B.keys()):
        B_results = results_by_B[B_val]
        best_result = max(B_results, key=lambda r: r['fidelity'])
        # Add info about how many attempts were made
        best_result['n_attempts'] = len(B_results)
        results.append(best_result)
        
        # Save a copy of the best monitor for easy access (backward compatibility)
        if n_init_sets > 1:
            try:
                B_dir = save_dir / f'B{int(B_val)}MHz'
                best_init_idx = best_result['init_idx']
                source_path = B_dir / f'monitor_init{best_init_idx}.pkl'
                best_path = B_dir / 'monitor_best.pkl'
                
                if source_path.exists():
                    # Load and save as best
                    import shutil
                    shutil.copy(source_path, best_path)
            except Exception as e:
                print(f"⚠️  Failed to save best monitor for B={B_val}: {str(e)}")
    
    if n_init_sets > 1:
        print(f"\n{'='*70}")
        print(f"📊 Multi-Start Optimization Summary:")
        print(f"{'='*70}")
        for result in results:
            print(f"  B = {result['Rydberg_B_MHz']:.0f} MHz: "
                  f"Best F = {result['fidelity']:.6f} from {result['n_attempts']} attempts")
        print(f"{'='*70}\n")
    
    # ========== Print Summary ==========
    print(f"\n{'='*70}")
    print(f"✅ Parallel Optimization Complete!")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Total optimizations run: {len(all_results)}/{n_tasks}")
    print(f"  Average per optimization: {total_time/len(all_results):.1f} seconds")
    print(f"  Successful B values: {len(results)}/{n_B_values}")
    print(f"  Success rate: {len(results)/n_B_values*100:.1f}%")
    print(f"  Theoretical speedup: ~{n_jobs}x")
    print(f"{'='*70}\n")
    
    # ========== Save Summary ==========
    if results:
        summary_data = [{
            'Rydberg_B_MHz': r['Rydberg_B_MHz'],
            'fidelity': r['fidelity'],
            **r['params']
        } for r in results]
        
        df = pd.DataFrame(summary_data)
        summary_path = save_dir / 'summary.csv'
        df.to_csv(summary_path, index=False)
        
        print(f"💾 Saved summary to: {summary_path}\n")
        print(df.to_string(index=False))
        print(f"\n{'='*70}")
        
        # Print best result
        best_idx = df['fidelity'].idxmax()
        print(f"\n🏆 Best Fidelity:")
        print(f"  B = {df.loc[best_idx, 'Rydberg_B_MHz']:.0f} MHz")
        print(f"  Fidelity = {df.loc[best_idx, 'fidelity']:.6f}")
        print(f"  T_gate = {df.loc[best_idx, 'T_gate']:.4f} μs")
        print(f"  tau_ratio = {df.loc[best_idx, 'tau_ratio']:.4f}")
        print(f"{'='*70}\n")
    else:
        print("⚠️  No successful optimizations!")
    
    return results


if __name__ == '__main__':
    # ========== Task Configuration ==========
    # Set a descriptive task name for this optimization run
    # This will be used to create a unique results folder
    task_name = 'CZ_gate_ARP_RydbergB'  # Modify this for different runs
    
    # Examples:
    # task_name = 'CZ_gate_quick_test'
    # task_name = 'CZ_gate_high_B_range'
    # task_name = 'CZ_gate_fine_sweep'
    
    # ========== Optimization Parameters ==========
    # Rydberg blockade strengths to optimize [MHz]
    # For full scan, use:
    # scale_B_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
    #                 1250, 1500, 1750, 2000, 2500, 3000]
    scale_B_list = [1000, 1250, 1500, 1750, 2000, 2500, 3000]
    # scale_B_list = [100]
    
    # Parameter bounds
    bounds = {
        'T_gate': (0.15, 1.5),
        'tau_ratio': (0.175, 0.175), # Fixed tau_ratio for ARP ( why? )
        'amp_Omega_r': (5*2*np.pi, 20*2*np.pi),
        'amp_Delta_r': (10*2*np.pi, 35*2*np.pi)
    }

    initial_params = {
        'T_gate': 0.9,
        'tau_ratio': 0.175,
        'amp_Omega_r': 90,
        'amp_Delta_r': 150
    }
    initial_params_list = [initial_params]

    # Generate several well-separated initial parameter sets
    # initial_params_list = generate_random_initial_params(bounds, n_samples=15, min_distance=0.3)
    # print(initial_params_list)
    
    try:
        results = main(scale_B_list, initial_params_list, bounds, task_name=task_name)
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user!")
        print("   Partial results may have been saved to optimization_results/")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
