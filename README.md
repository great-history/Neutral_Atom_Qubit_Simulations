# Neutral Atom Qubits Simulations

Python codes for simulating and optimizing quantum gates in neutral atom qubit systems using Rydberg interactions.

## Project Structure

```
Neutral_Atom_Qubit_Simulations/
├── myPkg/                               # Core Package (Utility Modules)
│   ├── atom_basis.py                    # Atomic basis state definitions (4-level system)
│   ├── hamiltonian_builder.py           # Time-dependent Hamiltonian construction
│   ├── pulse_functions.py               # Pulse shape functions (ARP, Gaussian, window)
│   ├── gates.py                         # Quantum gate definitions (X, Z, CZ)
│   ├── fidelity_calculator.py           # Gate fidelity computation (mixed/arithmetic/geometric)
│   ├── optimization_utils.py            # Optimization utilities and monitors
│   ├── plotting_helpers.py              # Visualization functions
│   └── utils.py                         # General utility functions
│
├── common_imports.py                    # Shared imports and configurations for all modules
│
├── module1_atom_exercises/              # Module 1: Atomic System Fundamentals
│   ├── module1_single_atom_dynamics.ipynb  # Four-level atom Hamiltonian evolution
│   └── module1_two_atom_coupling.ipynb     # Rydberg blockade and two-atom interactions
│
├── module2_fidelity_exercises/          # Module 2: Gate Fidelity Analysis
│   ├── module2_test_gate_fidelity.ipynb    # Fidelity metrics and computation methods
│   └── module2_test_gate_optimization.ipynb # Basic optimization techniques
│
├── module2_XGATE_exercises/             # Module 2: X-Gate (Single-Qubit π Rotation)
│   ├── Default_SQCONFIG_for_X_GATE.py      # Configuration file for X-gate parameters
│   ├── module2_X_gates_realization.ipynb   # X-gate with Gaussian pulses
│   ├── module2_X_gates_optimization.ipynb  # Optimize pulse width σ
│   └── save_data/                          # X-gate optimization results
│       └── XGate/
│
├── module2_ZGATE_exercises/             # Module 2: Z-Gate (Single-Qubit Phase Gate)
│   ├── Default_SQCONFIG_for_Z_GATE.py      # Configuration file for Z-gate parameters
│   ├── module2_Z_gate_realization.ipynb    # Z-gate with detuning pulses
│   ├── module2_Z_gate_optimization.ipynb   # Optimize gate time and detuning
│   └── save_data/                          # Z-gate optimization results
│       └── ZGate_Optimization/
│
├── module2_CZGATE_ARP_exercises/        # Module 2: CZ-Gate (Two-Qubit Controlled Phase, ARP)
│   ├── Default_TQCONFIG_for_CZ_GATE.py     # Configuration file for CZ-gate parameters
│   ├── module2_CZ_gate_ARP_realization.ipynb  # CZ gate with ARP protocol
│   ├── module2_CZ_gate_ARP_optimization.ipynb # Interactive parameter optimization
│   ├── module2_CZ_gate_fidelity_optimization_multiprocess.py  # Parallel optimization script
│   ├── module2_CZ_gate_fidelity_optimization_multiprocess_plotting.ipynb  # Results analysis
│   ├── images/                             # Plots and figures for CZ gate
│   └── save_data/                          # CZ-gate optimization results (various B values)
│
├── module2_CZGATE_TO_exercises/         # Module 2: CZ-Gate (Time-Optimal)
│   ├── Default_TQCONFIG_for_CZ_GATE_TO.py  # Configuration for time-optimal CZ gate
│   └── module2_CZ_gate_realization_time_optimal_pulse.ipynb  # Time-optimal pulses
│
├── save_data/                           # General optimization results and data
│   ├── 20260108_XXXXXX_CZ_gate_ARP_RydbergB/  # Timestamped CZ gate runs
│   ├── CZGate_ARP/                         # ARP protocol results
│   ├── optimization_results/               # Multi-parameter optimization data
│   └── XGate/                              # X-gate results
│
├── lecture_notes/                       # Course materials and references
├── show_version.ipynb                   # Python and package version information
├── README.md                            # This file
└── .gitignore
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

This project has been tested with the following environment:

- **Python**: 3.13.7
- **numpy**: 2.3.3
- **matplotlib**: 3.10.7
- **scipy**: 1.16.2
- **pandas**: 2.3.2
- **qutip**: 5.2.1

You can check your current versions by running [show_version.ipynb](show_version.ipynb).

### Setup
```bash
# Install dependencies with specific versions
pip install qutip==5.2.1 numpy==2.3.3 scipy==1.16.2 pandas==2.3.2 matplotlib==3.10.7

# Or install the latest compatible versions
pip install qutip numpy scipy pandas matplotlib

# Clone or download this repository
cd Neutral_Atom_Qubit_Simulations
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
1. Lecture Notes of Ph709 Quantum Computing Laboratory: Software Lab
2. Saffman, M., et al. "Symmetric Rydberg controlled-Z gates with adiabatic pulses." Physical Review A 101.6 (2020): 062309.
