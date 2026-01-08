from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def create_save_directory(task_name, base_dir='save_data', subdir='optimization_results'):
    """
    Create a timestamped directory for saving results.
    
    Parameters:
    -----------
    task_name : str
        Name of the task/experiment
    base_dir : str
        Base directory name (default: 'save_data')
    subdir : str
        Subdirectory name (default: 'optimization_results')
    
    Returns:
    --------
    Path
        Path object of the created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{timestamp}_{task_name}"
    save_dir = Path(base_dir) / subdir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Save directory created: {save_dir}")
    return save_dir


def load_optimization_summary(data_dir, verbose=True):
    """
    Load optimization summary data from CSV file.
    
    Parameters:
    -----------
    data_dir : str or Path
        Directory containing the optimization results
    verbose : bool
        Whether to print loading information (default: True)
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing summary data, or None if file not found
    """
    data_dir = Path(data_dir)
    summary_path = data_dir / 'summary.csv'
    
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if verbose:
            print(f"✅ Loaded summary data: {len(df)} optimization results\n")
            print(df)
        return df
    else:
        if verbose:
            print(f"❌ Summary file not found: {summary_path}")
            print("   Please run module2_CZ_gate_fidelity_optimization_multi_process.py first")
        return None


def save_optimization_summary(summary_data, save_dir, verbose=True):
    """
    Save optimization results to CSV file.
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing optimization results.
        Each dict should have 'Rydberg_B_MHz', 'fidelity', and 'params' keys.
    save_dir : str or Path
        Directory to save the summary CSV file
    verbose : bool
        Whether to print saving information (default: True)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the saved summary data
    """
    save_dir = Path(save_dir)
    
    # Convert to DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_path = save_dir / 'summary.csv'
    df.to_csv(summary_path, index=False)
    
    if verbose:
        print(f"💾 Saved summary to: {summary_path}")
        print(f"   Total results: {len(df)}\n")
        print(df.to_string(index=False))
    
    return df


def generate_random_initial_params(bounds, n_samples=10, min_distance=0.2, seed=None, max_attempts=1000):
    """
    Generate random initial parameter sets within specified bounds with minimum distance constraint.
    
    This function helps avoid local minima by providing multiple diverse
    starting points for optimization. It ensures that generated parameter sets
    are sufficiently different from each other to explore different regions
    of the parameter space.
    
    Effectiveness Note:
    -------------------
    Multi-start optimization with this function has shown significant improvements
    in practice. For example, in CZ gate optimization at B = 50 MHz:
    - Single fixed initial point: F = 0.973507 (infidelity ≈ 2.6×10⁻²)
    - Multi-start with random initial points: F = 0.999104 (infidelity ≈ 9×10⁻⁴)
    This represents a ~30× reduction in infidelity, demonstrating the effectiveness
    of exploring multiple starting points to escape local minima.
    
    Parameters
    ----------
    bounds : dict
        Dictionary of parameter bounds. Each key is a parameter name,
        and each value is a tuple (min, max).
        Example: {'T_gate': (0.25, 2.5), 'tau_ratio': (0.05, 0.75)}
    n_samples : int, optional
        Number of random initial parameter sets to generate.
        Default: 10
    min_distance : float, optional
        Minimum normalized Euclidean distance between any two parameter sets.
        Distance is computed in normalized space [0, 1] for each parameter.
        Typical values: 0.1 (closer), 0.2 (moderate), 0.3 (well-separated).
        Default: 0.2
    seed : int, optional
        Random seed for reproducibility. If None, results will be random.
        Default: None
    max_attempts : int, optional
        Maximum number of attempts to generate a valid parameter set
        that satisfies the distance constraint before giving up.
        Default: 1000
    
    Returns
    -------
    initial_params_list : list of dict
        List of randomly generated initial parameter dictionaries.
        Each dictionary has the same keys as the bounds input.
    
    Raises
    ------
    ValueError
        If unable to generate enough diverse parameter sets within max_attempts.
    
    Examples
    --------
    >>> bounds = {
    ...     'T_gate': (0.25, 2.5),
    ...     'tau_ratio': (0.05, 0.75),
    ...     'amp_Omega_r': (5*2*np.pi, 20*2*np.pi),
    ...     'amp_Delta_r': (10*2*np.pi, 30*2*np.pi)
    ... }
    >>> # Generate 5 well-separated parameter sets
    >>> initial_params_list = generate_random_initial_params(bounds, n_samples=5, min_distance=0.3)
    >>> len(initial_params_list)
    5
    """
    if seed is not None:
        np.random.seed(seed)
    
    initial_params_list = []
    param_names = list(bounds.keys())
    n_params = len(param_names)
    
    # Precompute parameter ranges for normalization
    param_ranges = {name: bounds[name][1] - bounds[name][0] for name in param_names}
    
    def normalize_params(params):
        """Normalize parameters to [0, 1] range for distance calculation."""
        normalized = np.zeros(n_params)
        for i, name in enumerate(param_names):
            min_val, max_val = bounds[name]
            # Handle fixed parameters (where param_range is 0)
            if param_ranges[name] == 0:
                normalized[i] = 0  # Fixed parameter doesn't contribute to distance
            else:
                normalized[i] = (params[name] - min_val) / param_ranges[name]
        return normalized
    
    def compute_min_distance(new_params_normalized, existing_params_normalized_list):
        """Compute minimum distance to existing parameter sets."""
        if not existing_params_normalized_list:
            return np.inf
        distances = [np.linalg.norm(new_params_normalized - existing) 
                     for existing in existing_params_normalized_list]
        return np.min(distances)
    
    # Store normalized versions for efficient distance computation
    normalized_params_list = []
    
    for i in range(n_samples):
        attempts = 0
        valid_params_found = False
        
        while attempts < max_attempts and not valid_params_found:
            # Generate random parameters
            params = {}
            for name in param_names:
                min_val, max_val = bounds[name]
                params[name] = np.random.uniform(min_val, max_val)
            
            # Normalize for distance calculation
            params_normalized = normalize_params(params)
            
            # Check distance constraint
            min_dist = compute_min_distance(params_normalized, normalized_params_list)
            
            if min_dist >= min_distance:
                initial_params_list.append(params)
                normalized_params_list.append(params_normalized)
                valid_params_found = True
            
            attempts += 1
        
        if not valid_params_found:
            raise ValueError(
                f"Unable to generate parameter set {i+1}/{n_samples} with "
                f"min_distance={min_distance} after {max_attempts} attempts. "
                f"Try reducing min_distance or increasing max_attempts."
            )
    
    return initial_params_list