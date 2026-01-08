# Neutral Atom Qubits Simulations

Some python codes  for simulating and optimizing quantum gates in neutral atom qubit systems using Rydberg interactions.

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

## TODO

- [ ] Refactor core modules into `myPkg` package
  - [ ] Move `atom_basis.py` → `myPkg/`
  - [ ] Move `hamiltonian_builder.py` → `myPkg/`
  - [ ] Move `pulse_functions.py` → `myPkg/`
  - [ ] Move `gates.py` → `myPkg/`
  - [ ] Move `fidelity_calculator.py` → `myPkg/`
  - [ ] Move `optimization_utils.py` → `myPkg/`
  - [ ] Move `plotting_helpers.py` → `myPkg/`
  - [ ] Move `utils.py` → `myPkg/`
  - [ ] Move `common_imports.py` → `myPkg/`
- [ ] Update import statements in all notebooks and scripts
- [ ] Add `__init__.py` to `myPkg` for package initialization
- [ ] Update documentation with new import paths

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
