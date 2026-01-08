# Neutral Atom Qubits Simulations

Python codes for simulating and optimizing quantum gates in neutral atom qubit systems using Rydberg interactions.

## Overview

This project provides tools for:
- **Single-qubit gates**: X-gate and Z-gate with Gaussian and window pulses
- **Two-qubit gates**: CZ gate with Adiabatic Rapid Passage (ARP) protocol
- **Gate fidelity optimization**: Multi-parameter optimization with parallel processing
- **Pulse engineering**: Custom pulse shapes for high-fidelity quantum operations
- **Visualization**: Comprehensive plotting utilities for pulse shapes, population dynamics, and optimization convergence

## Project Structure

```
Neutral_Atom_Qubit_Simulations/
├── myPkg/                               # Core Package
│   ├── atom_basis.py                    # Atomic basis state definitions (4-level system)
│   ├── hamiltonian_builder.py           # Time-dependent Hamiltonian construction
│   ├── pulse_functions.py               # Pulse shape functions (ARP, Gaussian, window)
│   ├── gates.py                         # Quantum gate definitions (X, Z, CZ)
│   ├── fidelity_calculator.py           # Gate fidelity computation (mixed/arithmetic/geometric)
│   ├── optimization_utils.py            # Optimization utilities and monitors
│   ├── plotting_helpers.py              # Visualization functions
│   └── utils.py                         # General utility functions
│
├── Configuration Files
│   ├── common_imports.py                # Common imports and configurations
│   ├── Default_SQCONFIG_for_X_GATE.py   # Single-qubit X gate default config
│   ├── Default_SQCONFIG_for_Z_GATE.py   # Single-qubit Z gate default config
│   └── Default_TQCONFIG_for_CZ_GATE.py  # Two-qubit CZ gate default config
│
├── Module 1: Single & Two-Atom Dynamics
│   ├── module1_single_atom_dynamics.ipynb
│   └── module1_two_atom_coupling.ipynb
│
├── Module 2: Gate Optimization & Realization
│   ├── Exercise 1 - X Gate (Single-Qubit)
│   │   ├── module2_X_gates_realization.ipynb      # X-gate with Gaussian pulses
│   │   └── module2_X_gates_optimization.ipynb     # Optimize pulse width σ
│   │
│   ├── Exercise 2 - Z Gate (Single-Qubit)
│   │   ├── module2_Z_gate_realization.ipynb       # Z-gate with detuning pulses
│   │   └── module2_Z_gate_optimization.ipynb      # Optimize gate time and detuning
│   │
│   └── Exercise 3 - CZ Gate (Two-Qubit)
│       ├── module2_CZ_gate_ARP_realization.ipynb  # CZ gate with ARP protocol
│       ├── module2_CZ_gate_ARP_optimization.ipynb # Parameter optimization
│       ├── module2_CZ_gate_fidelity_optimization_multi_process.py
│       ├── module2_CZ_gate_fidelity_optimization_multi_process_plotting.ipynb
│       ├── module2_CZ_gate_realization_time_optimal_pulse.ipynb
│       ├── module2_test_gate_fidelity.ipynb
│       └── module2_test_gate_optimization.ipynb
│
└── Data & Resources
    ├── save_data/                       # Saved optimization data and results
    │   ├── XGate/                       # X-gate optimization results
    │   ├── ZGate_Optimization/          # Z-gate optimization results
    │   ├── CZGate_ARP/                  # CZ-gate ARP optimization results
    │   └── optimization_results/        # Other optimization data
    ├── images/                          # Figures, diagrams, and screenshots
    └── lecture_notes/                   # Course materials and references
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

### Units Convention
- **Time**: microseconds (μs)
- **Frequency**: MHz for display, rad/μs (angular frequency) for computation
- **Conversion**: $f$ (MHz) = $\omega$ (rad/μs) / (2π)

## Module Descriptions

### Module 1: Fundamentals
- **Single atom dynamics**: Four-level system Hamiltonian evolution and population dynamics
- **Two-atom coupling**: Rydberg blockade interactions and entanglement generation

### Module 2: Quantum Gate Implementation

#### Exercise 1: X-Gate (Single-Qubit π Rotation)
**Implementation Method:**
- Gaussian pulse: $\Omega_{01}(t) = \Omega_0 \exp[-(t-t_0)^2/(2\sigma^2)]$
- Direct drive on computational basis $|0\rangle \leftrightarrow |1\rangle$
- Set $\Omega_r = 0, \Delta_r = 0$ (no Rydberg coupling)

**Key Tasks:**
1. Scan Rabi frequency $\Omega_0$ and pulse width $\sigma$
2. Optimize $\sigma$ for fixed $\Omega_0$ to achieve π-pulse
3. Verify analytic formula: $\sigma \cdot \Omega_0 = \sqrt{\pi/2}$
4. Visualize Bloch sphere evolution

**Expected Results:**
- Minimal leakage to $|r\rangle$ and $|d\rangle$ states ($< 10^{-12}$)
- High fidelity ($> 0.999$) with optimized parameters
- Clean population transfer $|0\rangle \leftrightarrow |1\rangle$

#### Exercise 2: Z-Gate (Single-Qubit Phase Gate)
**Implementation Method:**
- Window (square) pulse on detuning: $\delta_1(t)$ during $0 < t < T$
- Pure phase accumulation, no population transfer
- Set $\Omega_{01} = 0, \Omega_r = 0, \Delta_r = 0$

**Key Tasks:**
1. Scan detuning amplitude $\delta_1$ and gate time $T$
2. Optimize to achieve phase $\phi = \int_0^T \delta_1(t) dt = \pi$
3. Plot fidelity vs parameters
4. Visualize state trajectory on Bloch sphere (rotation around Z-axis)

**Expected Results:**
- No leakage to auxiliary states
- Constraint: $\delta_1 \times T = \pi$ for optimal Z-gate
- Near-perfect fidelity ($F \approx 1$) at optimal parameters

#### Exercise 3: CZ-Gate (Two-Qubit Controlled Phase)
**Implementation Method:**
- Adiabatic Rapid Passage (ARP) protocol (Ref. [2])
- Time-dependent Rabi drive: $\Omega_{1r}(t) = \Omega_{1r}[e^{-(t-t_0)^4/\tau^4} - a]/(1-a)$
- Time-dependent detuning: $\Delta_r(t) = -\Delta_r \cos(2\pi t/T)$
- Rydberg blockade strength: $B/2\pi = 50-3000$ MHz

**Key Tasks:**
1. Reproduce Ref. [2] Fig. 2 with initial parameters
2. Analyze population dynamics for $|1,0\rangle$ and $|1,1\rangle$ initial states
3. Scan Rydberg lifetime $\gamma_r$ and measure fidelity dependence
4. Optimize parameters for various blockade strengths $B$
5. Compare three parameter sets (slow/reference/fast gates)

**Key Observations:**
- **Significant Rydberg population**: Unlike X/Z gates, CZ requires transient $|r\rangle$ population ($\sim 10^{-4}$)
- **Leakage to decay state**: $|d\rangle$ population $\sim 10^{-3}$ due to spontaneous emission
- **Lifetime sensitivity**: Fidelity strongly depends on $\gamma_r$ (saturates below $1/540$ μs⁻¹)
- **Blockade scaling**: Requires $B \gtrsim 200$ MHz for near-optimal fidelity

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
