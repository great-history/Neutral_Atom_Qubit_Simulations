from pathlib import Path
from datetime import datetime
import pandas as pd


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
