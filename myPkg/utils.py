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
    if subdir is None or subdir == '':
        save_dir = Path(base_dir) / folder_name
    else:
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


def save_optimization_summary(summary_data, save_dir, filename='summary.csv', verbose=True):
    """
    Save optimization results to a CSV file.

    Parameters:
    -----------
    summary_data : list | dict | pd.DataFrame
        Data to be saved. Commonly a list of dictionaries containing optimization results.
    save_dir : str or Path
        Directory to save the summary CSV file.
    filename : str or Path, optional
        Output CSV filename. Default: 'summary.csv'.
        Use this to avoid overwriting when saving multiple DataFrames into the same folder.
        Examples: 'summary_run1.csv', 'B_sweep_summary.csv'
    verbose : bool
        Whether to print saving information (default: True)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the saved summary data
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = summary_data if isinstance(summary_data, pd.DataFrame) else pd.DataFrame(summary_data)

    # Build output path (force .csv if user didn't provide it)
    filename = Path(filename)
    if filename.suffix.lower() != ".csv":
        filename = filename.with_suffix(".csv")
    summary_path = save_dir / filename.name

    # Save to CSV
    df.to_csv(summary_path, index=False)

    if verbose:
        print(f"💾 Saved summary to: {summary_path}")
        print(f"   Total results: {len(df)}\n")
        print(df.to_string(index=False))

    return df


def load_multiple_optimization_summaries(data_dirs, add_source_column=True, 
                                         drop_duplicates_by=None, keep_duplicate='first',
                                         sort_by=None, ascending=True, verbose=True):
    """
    Concatenate summary.csv files from multiple data directories into a single DataFrame.
    
    This function is useful for combining optimization results from multiple runs
    or different parameter sweeps into one unified dataset for analysis and comparison.
    
    Parameters
    ----------
    data_dirs : list of str or Path
        List of directory paths containing summary.csv files.
        Each directory should have a 'summary.csv' file in its root.
    add_source_column : bool, optional
        If True, adds a 'source_dir' column to track which directory each row came from.
        This is helpful for identifying the origin of each result.
        Default: True
    drop_duplicates_by : str or list of str, optional
        Column name(s) to use for identifying duplicate rows. If specified, duplicate
        rows based on these columns will be removed. Common use: 'Rydberg_B_MHz' to
        keep only one result per B value. If None, no deduplication is performed.
        Default: None
    keep_duplicate : {'first', 'last', False}, optional
        Which duplicate to keep when drop_duplicates_by is specified.
        - 'first': Keep the first occurrence
        - 'last': Keep the last occurrence
        - False: Drop all duplicates
        Default: 'first'
    sort_by : str or list of str, optional
        Column name(s) to sort the DataFrame by. If None, no sorting is performed.
        Common use: 'Rydberg_B_MHz' to sort by B value.
        Default: None
    ascending : bool or list of bool, optional
        Sort order for sort_by. If True, sort in ascending order. If False, sort in
        descending order. If a list, must match the length of sort_by.
        Default: True
    verbose : bool, optional
        If True, prints loading information and summary statistics.
        Default: True
    
    Returns
    -------
    pd.DataFrame or None
        Concatenated DataFrame containing all summary data from the specified directories.
        Returns None if no valid summary files are found.
    
    Examples
    --------
    >>> # Basic concatenation
    >>> data_dirs = [
    ...     'save_data/20260112_094421_CZ_gate_ARP_RydbergB',
    ...     'save_data/20260112_103940_CZ_gate_pulse_optimization'
    ... ]
    >>> combined_df = load_multiple_optimization_summaries(data_dirs)
    
    >>> # Remove duplicates based on Rydberg_B_MHz, keep best fidelity (last)
    >>> combined_df = load_multiple_optimization_summaries(
    ...     data_dirs, 
    ...     drop_duplicates_by='Rydberg_B_MHz', 
    ...     keep_duplicate='last'
    ... )
    
    >>> # Sort by B value and remove duplicates
    >>> combined_df = load_multiple_optimization_summaries(
    ...     data_dirs,
    ...     drop_duplicates_by='Rydberg_B_MHz',
    ...     sort_by='Rydberg_B_MHz',
    ...     ascending=True
    ... )
    
    >>> # Sort by fidelity in descending order
    >>> combined_df = load_multiple_optimization_summaries(
    ...     data_dirs,
    ...     sort_by='fidelity',
    ...     ascending=False
    ... )
    """
    df_list = []
    successful_dirs = []
    failed_dirs = []
    
    if verbose:
        print(f"{'='*70}")
        print(f"Concatenating summary files from {len(data_dirs)} directories")
        print(f"{'='*70}\n")
    
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        summary_path = data_dir / 'summary.csv'
        
        if summary_path.exists():
            try:
                df = pd.read_csv(summary_path)
                
                # Add source directory information if requested
                if add_source_column:
                    df['source_dir'] = data_dir.name
                
                df_list.append(df)
                successful_dirs.append(data_dir.name)
                
                if verbose:
                    print(f"✅ Loaded: {data_dir.name}")
                    print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
                
            except Exception as e:
                failed_dirs.append((data_dir.name, str(e)))
                if verbose:
                    print(f"❌ Error loading {data_dir.name}: {e}")
        else:
            failed_dirs.append((data_dir.name, "summary.csv not found"))
            if verbose:
                print(f"⚠️  Not found: {summary_path}")
    
    if verbose:
        print(f"\n{'-'*70}")
        print(f"Summary:")
        print(f"  Successfully loaded: {len(successful_dirs)}/{len(data_dirs)}")
        print(f"  Failed: {len(failed_dirs)}/{len(data_dirs)}")
        print(f"{'-'*70}\n")
    
    if not df_list:
        if verbose:
            print("❌ No valid summary files found!")
        return None
    
    # Concatenate all DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Remove duplicates if specified
    if drop_duplicates_by is not None:
        rows_before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=drop_duplicates_by, keep=keep_duplicate)
        rows_after = len(combined_df)
        
        if verbose:
            print(f"🔄 Deduplication:")
            print(f"   Column(s): {drop_duplicates_by}")
            print(f"   Keep: {keep_duplicate}")
            print(f"   Removed {rows_before - rows_after} duplicate rows")
            print(f"   Remaining: {rows_after} rows\n")
    
    # Sort if specified
    if sort_by is not None:
        combined_df = combined_df.sort_values(by=sort_by, ascending=ascending)
        combined_df = combined_df.reset_index(drop=True)
        
        if verbose:
            print(f"🔀 Sorting:")
            print(f"   Column(s): {sort_by}")
            print(f"   Order: {'Ascending' if ascending else 'Descending'}\n")
    
    if verbose:
        print(f"✅ Combined DataFrame:")
        print(f"   Total rows: {len(combined_df)}")
        print(f"   Total columns: {len(combined_df.columns)}")
        print(f"   Columns: {list(combined_df.columns)}\n")
        
        if add_source_column and 'source_dir' in combined_df.columns:
            print(f"Distribution by source directory:")
            print(combined_df['source_dir'].value_counts().to_string())
            print()
        
        print(f"{'-'*70}")
        print(combined_df.head(10))
        print(f"{'-'*70}\n")
    
    return combined_df


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