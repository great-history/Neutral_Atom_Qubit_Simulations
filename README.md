# Neutral Atom Qubits Simulations

A comprehensive Python framework for simulating and optimizing quantum gates in neutral atom qubit systems using Rydberg interactions.

## Overview

This project provides tools for:
- **Single-qubit gates**: X, Y, Z gates with Gaussian and window pulses
- **Two-qubit gates**: CZ gate with Adiabatic Rapid Passage (ARP) and time-optimal pulses
- **Gate fidelity optimization**: Multi-parameter optimization with parallel processing
- **Pulse engineering**: Custom pulse shapes for high-fidelity quantum operations
- **Visualization**: Comprehensive plotting utilities for pulse shapes, population dynamics, and optimization convergence

## Project Structure

```
simulation_codes/
├── Core Modules
│   ├── atom_basis.py                    # Atomic basis state definitions
│   ├── hamiltonian_builder.py           # Time-dependent Hamiltonian construction
│   ├── pulse_functions.py               # Pulse shape functions (ARP, Gaussian, etc.)
│   ├── gates.py                         # Quantum gate definitions
│   ├── fidelity_calculator.py           # Gate fidelity computation
│   ├── optimization_utils.py            # Optimization utilities and monitors
│   ├── plotting_helpers.py              # Visualization functions
│   ├── utils.py                         # General utility functions
│   └── common_imports.py                # Common imports and configurations
│
├── Configuration Files
│   ├── Default_SQCONFIG_for_X_GATE.py   # Single-qubit X gate default config
│   ├── Default_SQCONFIG_for_Z_GATE.py   # Single-qubit Z gate default config
│   └── Default_TQCONFIG_for_CZ_GATE.py  # Two-qubit CZ gate default config
│
├── Module 1: Single & Two-Atom Dynamics
│   ├── module1_single_atom_dynamics.ipynb
│   └── module1_two_atom_coupling.ipynb
│
├── Module 2: Gate Optimization & Realization
│   ├── Single-Qubit Gates
│   │   ├── module2_X_gates_optimization.ipynb
│   │   ├── module2_X_gates_realization.ipynb
│   │   ├── module2-1_X_Gate_Realization.ipynb
│   │   ├── module2_Z_gate_optimization.ipynb
│   │   └── module2_Z_gate_realization.ipynb
│   │
│   └── Two-Qubit CZ Gate
│       ├── module2_CZ_gate_ARP_optimization.ipynb
│       ├── module2_CZ_gate_ARP_realization.ipynb
│       ├── module2_CZ_gate_fidelity_optimization_multi_process.py
│       ├── module2_CZ_gate_fidelity_optimization_multi_process_plotting.ipynb
│       ├── module2_CZ_gate_realization_time_optimal_pulse.ipynb
│       ├── module2_test_gate_fidelity.ipynb
│       └── module2_test_gate_optimization.ipynb
│
└── Data Directories
    ├── optimization_results/            # Optimization results from multi-process runs
    ├── save_data/                       # Saved optimization data
    ├── images/                          # Figures and diagrams
    └── myPkg/                           # Custom package modules
```

## Key Features

### 1. Hamiltonian Construction
- Time-dependent single-atom Hamiltonians
- Two-atom Hamiltonians with Rydberg blockade coupling
- Lindblad dissipation operators for realistic decoherence modeling

### 2. Pulse Engineering
- **Adiabatic Rapid Passage (ARP)**: High-fidelity two-qubit gates
- **Gaussian pulses**: Single-qubit rotations
- **Window pulses**: Detuning-based single-qubit gates
- **Time-optimal pulses**: Fast gate operations with minimal error

### 3. Gate Fidelity Optimization
- **Single-qubit optimization**: `create_SQ_pulse_optimizer()`
- **Two-qubit optimization**: `create_TQ_pulse_optimizer()`
- **Multi-process parallelization**: Efficient parameter sweeps using `joblib`
- **Real-time monitoring**: `OptimizationMonitor` class for tracking convergence

### 4. Fidelity Metrics
- **Mixed fidelity**: Average over computational basis states
- **Geometric mean fidelity**: Product-based metric
- **Arithmetic mean fidelity**: Standard average fidelity

### 5. Visualization Tools
- Pulse shape plotting with automatic unit conversion
- Population dynamics with multi-panel layouts
- Optimization convergence tracking
- Parameter sweep visualization

## Installation

### Requirements
- Python 3.8+
- QuTiP (Quantum Toolbox in Python)
- NumPy, SciPy, Pandas
- Matplotlib
- Joblib (for parallel processing)

### Setup
```bash
# Install dependencies
pip install qutip numpy scipy pandas matplotlib joblib

# Clone or download this repository
cd simulation_codes
```

## Usage

### Quick Start: Single-Qubit X Gate Optimization

```python
from optimization_utils import create_SQ_pulse_optimizer, OptimizationMonitor
from pulse_functions import gaussian_pulse
from scipy.optimize import minimize
import numpy as np

# Setup parameters
pulse_funcs = {'Omega_r': gaussian_pulse}
atom_params = {'Omega_01': 0, 'delta_1': 0}
lindblad_params = {'gamma_r': 1/540, 'b_0r': 1/16, 'b_1r': 1/16, 'b_dr': 7/8}

# Create optimizer
objective = create_SQ_pulse_optimizer(
    pulse_type='Gaussian',
    pulse_functions=pulse_funcs,
    atom_base_params=atom_params,
    lindblad_params=lindblad_params,
    target_gate=X_gate,
    qs0_list=qs0_list,
    psi0_list=psi0_list,
    comp_indices=[0, 1],
    fidelity_type='mixed'
)

# Optimize with monitoring
param_names = ['sigma', 't0_ratio', 'amp_Omega_01']
monitor = OptimizationMonitor(param_names, objective)
x0 = [0.1, 5.0, 20*2*np.pi]  # Initial guess

result = minimize(objective, x0, method='Nelder-Mead', callback=monitor)
monitor.print_summary()
monitor.plot_convergence(save_path='results/convergence.png')
```

### Multi-Process CZ Gate Optimization

```bash
# Run parallel optimization across multiple Rydberg blockade strengths
python module2_CZ_gate_fidelity_optimization_multi_process.py
```

This will:
1. Optimize CZ gate fidelity for multiple B values (50-3000 MHz)
2. Use parallel processing to speed up computation
3. Save individual monitors and summary CSV to `optimization_results/`
4. Display real-time progress and best results

### Analyzing Results

```python
from utils import load_optimization_summary, create_save_directory
from pathlib import Path

# Load results
data_dir = Path('optimization_results/20260107_233700_CZ_gate_ARP_RydbergB')
df = load_optimization_summary(data_dir)

# Find best result
best_idx = df['fidelity'].idxmax()
print(f"Best fidelity: {df.loc[best_idx, 'fidelity']:.6f}")
print(f"At B = {df.loc[best_idx, 'Rydberg_B_MHz']:.0f} MHz")
```

## Physical System

### Atomic Level Structure
```
|d⟩ (decay state)
|r⟩ (Rydberg state) ← Ω_r(t), Δ_r(t)
|1⟩ (excited state) ← Ω_01, δ_1
|0⟩ (ground state)
```

### Key Parameters
- **Rabi frequencies**: Ω_r (ground to Rydberg), Ω_01 (ground to excited)
- **Detunings**: Δ_r (Rydberg detuning), δ_1 (excited state detuning)
- **Rydberg blockade**: B (MHz) - interaction strength between Rydberg atoms
- **Lindblad parameters**: γ_r (decay rate), b_0r, b_1r, b_dr (branching ratios)

### Units Convention
- **Time**: microseconds (μs)
- **Frequency**: MHz for display, rad/μs (angular frequency) for computation
- **Conversion**: f (MHz) = ω (rad/μs) / (2π)

## Module Descriptions

### Module 1: Fundamentals
- **Single atom dynamics**: Basic Hamiltonian evolution and population dynamics
- **Two-atom coupling**: Rydberg blockade interactions and entanglement

### Module 2: Gate Implementation
- **X gates**: π-rotation around X-axis using Gaussian pulses
- **Z gates**: Phase gates using detuning window pulses
- **CZ gate**: Controlled-Z gate with ARP or time-optimal pulses

## Optimization Workflow

1. **Define target gate**: Unitary matrix for desired quantum operation
2. **Set initial parameters**: Starting guess for pulse parameters
3. **Configure bounds**: Physical constraints on optimization variables
4. **Run optimization**: Use `scipy.optimize.minimize` with monitoring
5. **Analyze results**: Plot convergence, parameter evolution, and final fidelity
6. **Verify gate**: Test optimized parameters in quantum simulation

## Performance Tips

### For Multi-Process Optimization
- Adjust `n_jobs` based on CPU cores available
- Use `pre_dispatch` to control memory usage
- Enable result caching to skip completed optimizations
- Monitor disk space for saving large datasets

### For Single Optimization
- Start with coarse parameter sweeps
- Use `verbose=True` in monitor for real-time feedback
- Save monitors periodically for long runs
- Adjust `maxiter` and tolerances based on convergence

## Data Management

### Saving Results
```python
# Create timestamped directory
save_dir = create_save_directory('my_optimization')

# Save monitor
monitor.save(save_dir / 'monitor.pkl')

# Save summary
df.to_csv(save_dir / 'summary.csv', index=False)
```

### Loading Results
```python
# Load monitor (includes full history)
monitor = OptimizationMonitor.load('results/monitor.pkl')

# Load summary (quick overview)
df = load_optimization_summary('results/')
```

## Visualization Examples

### Pulse Shapes
```python
from plotting_helpers import plot_pulse_shapes

pulse_dict = {
    'Omega_r': {'data': APR_pulse_Omega_r, 'args': {...}, 'label': r'$\Omega_r$(t)'},
    'Delta_r': {'data': APR_pulse_Delta_r, 'args': {...}, 'label': r'$\Delta_r$(t)'}
}

plot_pulse_shapes(ax, tlist, pulse_dict, normalize_2pi=True)
```

### Population Dynamics
```python
from plotting_helpers import plot_population_evolution

plot_population_evolution(ax, tlist, pop_list, legend_list, 
                          xlabel='Time (μs)', ylabel='Population')
```

### Optimization Convergence
```python
monitor.plot_convergence(figsize=(12, 8), save_path='convergence.png')
```

## Contact

sfang65@wisc.edu

## Acknowledgments

This project uses:
- **QuTiP**: Quantum dynamics simulation
- **SciPy**: Numerical optimization
- **Joblib**: Parallel processing
- **Matplotlib**: Visualization

## References
1. Lecture Notes of Ph709 Quantum Computing Laboratory : Software Lab
2. Saffman, M., et al. "Symmetric Rydberg controlled-Z gates with adiabatic pulses." Physical Review A 101.6 (2020): 062309.
