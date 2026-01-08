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

# Import custom modules
from common_imports import *
# from ...pulse_functions import *
from optimization_utils import *


def optimize_for_single_B(idx, scale_B, Rydberg_B, initial_params, bounds, save_dir_str,
                          atom0_ham_params, atom1_ham_params, lindblad_params,
                          target_gate, qs0_list, psi0_list, comp_indices, total_count):
    """
    Optimize gate fidelity for a single Rydberg blockade strength.
    
    Parameters
    ----------
    idx : int
        Index in the B value list
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
    total_count : int
        Total number of B values to optimize
        
    Returns
    -------
    dict
        Optimization results including fidelity and optimal parameters
    """
    # Convert string path back to Path object (important for Windows multiprocessing)
    save_dir = Path(save_dir_str)
    
    # Check if results already exist (optional: enable caching)
    save_path = save_dir / f'monitor_B{int(scale_B)}MHz.pkl'
    # Uncomment to enable caching:
    # if save_path.exists():
    #     print(f"📂 [{idx+1}/{total_count}] Loading existing B = {scale_B:.0f} MHz")
    #     monitor = OptimizationMonitor.load(save_path)
    #     result = monitor.get_best_result()
    #     result['Rydberg_B_MHz'] = scale_B
    #     return result
    
    print(f"🚀 [{idx+1}/{total_count}] Starting B = {scale_B:.0f} MHz (PID: {os.getpid()})")
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
        num_time_points=300
    )
    
    # Optimization parameters
    param_names = ['T_gate', 'tau_ratio', 'amp_Omega_r', 'amp_Delta_r']
    x0 = np.array([initial_params[name] for name in param_names])
    param_bounds = [bounds[name] for name in param_names]
    options = {'maxiter': 300, 'disp': False, 'fatol': 1e-6, 'xatol': 1e-4}
    
    # Run optimization
    monitor = OptimizationMonitor(param_names, objective_func, verbose=False)
    try:
        result = minimize(objective_func, x0, method='Nelder-Mead',
                         bounds=param_bounds, options=options, callback=monitor)
    except Exception as e:
        print(f"❌ [{idx+1}/{total_count}] Error for B = {scale_B:.0f} MHz: {str(e)}")
        return None
    
    # Get best result
    best = monitor.get_best_result()
    best['Rydberg_B_MHz'] = scale_B
    
    # Save results
    try:
        # Ensure directory exists (important for multiprocessing on Windows)
        save_dir.mkdir(parents=True, exist_ok=True)
        monitor.save(save_path)
    except Exception as e:
        print(f"⚠️  [{idx+1}/{total_count}] Failed to save monitor: {str(e)}")
    
    elapsed = time.time() - start_time
    print(f"✅ [{idx+1}/{total_count}] Completed B = {scale_B:.0f} MHz, "
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


def main(task_name='CZ_gate_optimization_multi_process'):
    """
    Main function to run parallel optimization.
    
    Parameters
    ----------
    task_name : str
        Name of the optimization task. Used to create a unique results folder.
        Default: 'CZ_gate_optimization_multi_process'
    
    Returns
    -------
    results : list
        List of optimization results for each B value
    """
    
    # ========== Configuration ==========
    # Rydberg blockade strengths to optimize
    # scale_B_list = [400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000]  # [MHz]
    # For full scan, use:
    # scale_B_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
    #                 1250, 1500, 1750, 2000, 2500, 3000]
    scale_B_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    Rydberg_B_list = [scale_B * 2 * np.pi for scale_B in scale_B_list]
    
    # Initial parameters for optimization
    initial_params = {
        'T_gate': 0.4548,
        'tau_ratio': 0.1576,
        'amp_Omega_r': 62.83341,
        'amp_Delta_r': 110.7900
    }
    
    # Parameter bounds
    bounds = {
        'T_gate': (0.25, 2.5),
        'tau_ratio': (0.05, 0.75),
        'amp_Omega_r': (5*2*np.pi, 20*2*np.pi),
        'amp_Delta_r': (10*2*np.pi, 30*2*np.pi)
    }
    
    # ========== Optimized Parallel Configuration ==========
    # Dynamically determine optimal number of parallel jobs
    n_cores = cpu_count()
    n_tasks = len(Rydberg_B_list)
    
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
    save_dir = Path(__file__).parent / 'optimization_results' / folder_name
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
    print(f"  Total B values: {n_tasks}")
    print(f"  B range: {scale_B_list[0]:.0f} - {scale_B_list[-1]:.0f} MHz")
    print(f"  Parallel jobs: {n_jobs} (optimized)")
    print(f"  Pre-dispatch: {pre_dispatch}")
    print(f"  Backend: loky (optimized for Windows)")
    print(f"  Max iterations per B: 300")
    print(f"{'='*70}\n")
    
    total_start = time.time()
    
    # Run parallel optimization using joblib
    # verbose=10 shows progress bar, pre_dispatch controls task distribution
    results = Parallel(
        n_jobs=n_jobs,
        backend='loky',
        verbose=10,
        pre_dispatch=pre_dispatch,
        timeout=None  # No timeout for long optimization tasks
    )(
        delayed(optimize_for_single_B)(
            idx, scale_B, B, initial_params, bounds, str(save_dir),
            atom0_ham_params, atom1_ham_params, lindblad_params,
            target_gate, qs0_list, psi0_list, comp_indices, n_tasks
        )
        for idx, (scale_B, B) in enumerate(zip(scale_B_list, Rydberg_B_list))
    )
    
    total_time = time.time() - total_start
    
    # Filter out None results (failed optimizations)
    results = [r for r in results if r is not None]
    
    # ========== Print Summary ==========
    print(f"\n{'='*70}")
    print(f"✅ Parallel Optimization Complete!")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Average per B: {total_time/n_tasks:.1f} seconds")
    print(f"  Successful: {len(results)}/{n_tasks}")
    print(f"  Efficiency: {len(results)/n_tasks*100:.1f}%")
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
    
    try:
        results = main(task_name=task_name)
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user!")
        print("   Partial results may have been saved to optimization_results/")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
