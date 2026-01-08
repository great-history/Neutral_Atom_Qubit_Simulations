import math
from typing import Iterable, Mapping, Optional, Sequence
# from qutip import QObj

import matplotlib.pyplot as plt
import numpy as np

# Simple helpers for common plots used across the notebooks.

# Global matplotlib configuration dictionary
plt_config = {
    # 'figure.dpi': 200,
    'savefig.dpi': 300,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans'],
    'mathtext.fontset': 'dejavusans',  # 'dejavusans', 'stix', 'cm' (Computer Modern)
    'axes.unicode_minus': False,  # Use proper minus sign
}

# Global plot style configuration
plotting_styles = {
    'Omega_r_pulse': {
        'color': 'blue',
        'marker': '>',
        'markersize': 5,
        'markevery': 2,
        'linewidth': 0.75,
        'alpha': 0.8
    },
    'Delta_r_pulse': {
        'color': 'red',
        'marker': 'o',
        'markersize': 5,
        'markevery': 2,
        'linewidth': 0.75,
        'alpha': 0.8
    },
    'Detuning_pulse': {
        'color': 'gray',
        'linestyle': '--',
        'linewidth': 1.0,
        'alpha': 0.7
    },
    'Gaussian_pulse': {
        'color': 'black',
        'linestyle': ':',
        'linewidth': 1.0,
        'alpha': 0.7
    },
    'pop_style_a': {
        'linewidth': 1,
    },
    'pop_style_b': {
        'linewidth': 1,
        'linestyle': '--',
        # 'alpha': 0.6,
    },
    'fidelity_style_0': {
        'linestyle': '-',
        'linewidth': 1.0,
    },
    'fidelity_style_1': {
        'linestyle': '--',
        'linewidth': 1.0,
    },
    'fidelity_style_a': {
        'color': 'green',
        'marker': 's',  # square
        'markersize': 7.5,
        'linestyle': '--',
        'linewidth': 1.0,
        'alpha': 0.75,
    },
    'fidelity_style_b': {
        'color': 'blue',
        'marker': 'o',
        'markersize': 7.5,
        'linestyle': '-',
        'linewidth': 1.0,
        'alpha': 0.75,
    },
    'fidelity_style_c': {
        'color': 'brown',
        'marker': '^',  # triangle
        'markersize': 7.5,
        'linestyle': ':',
        'linewidth': 1.0,
        'alpha': 0.75,
    },
    'fidelity_style_d': {
        'color': 'purple',
        'marker': 'd', 
        'markersize': 7.5,
        'linestyle': ':',
        'linewidth': 1.0,
        'alpha': 0.75,
    },
}

# Legend style configuration
legend_styles = {
    'default': {
        'loc': 'upper right',
        'ncol': 3,
        'frameon': False,
        'fontsize': 10
    },
    'compact': {
        'loc': 'best',
        'ncol': 1,
        'frameon': False,
        'fontsize': 9
    },
    'horizontal': {
        'loc': 'upper center',
        'ncol': 4,
        'frameon': False,
        'fontsize': 9
    },
    'outside_left': {
        'bbox_to_anchor': (1.05, 1),
        'loc': 'upper left',
        'ncol': 1,
        'frameon': False,
        'fontsize': 9,
        'fancybox': False
    },
}


def plot_population_evolution(
    ax: plt.Axes,
    tlist: Sequence[float],
    pop_list: Sequence[np.ndarray],
    legend_list: Sequence[str],
    plot_pulse: bool = False,
    pulse_dict: dict = {},
    title: Optional[str] = None,
    xlabel: str = r"Time ($\mu$s)",
    ylabel: str = "Population",
    log_scale: bool = False,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    grid_alpha: float = 0.3,
    show_legend: bool = True,
    plotting_style_list: Optional[Sequence[dict]] = None,
) -> plt.Axes:
    """
    Plot population evolution on a single axes without the use of QuTiP's built-in plotting functions.
    
    Parameters:
    ----------
    ax : plt.Axes
        The axes to plot on
    tlist : Sequence[float]
        Time points
    pop_list : Sequence[np.ndarray]
        List of population data arrays to plot
    legend_list : Sequence[str]
        List of labels for each population curve
    pulse_shape : np.ndarray, optional
        Pulse shape data to overlay on the plot (if plot_pulse is True)
    pulse_label : str
        Label for the pulse shape curve
    plot_pulse : bool
        Whether to plot the pulse shape
    pulse_style : dict, optional
        Style dictionary for the pulse shape curve (e.g., plotting_styles['Gaussian_pulse'])
    title : str, optional
        Title for the axes
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    log_scale : bool
        If True, plot log10 of populations
    xlim : tuple, optional
        X-axis limits as (xmin, xmax)
    ylim : tuple, optional
        Y-axis limits as (ymin, ymax)
    grid_alpha : float
        Grid transparency
    show_legend : bool
        Whether to show legend on this axes
    plotting_style_list : Sequence[dict], optional
        List of style dictionaries for each population curve.
        If None, uses default styles with different linestyles
    
    Returns:
    -------
    ax : plt.Axes
        The modified axes object
    
    Example:
    -------
    >>> fig, ax = plt.subplots()
    >>> pop_list = [result.expect[0], result.expect[1]]
    >>> legend_list = [r"pop of $|1\rangle$", r"pop of $|r\rangle$"]
    >>> plot_population_evolution(ax, tlist, pop_list, legend_list, \
                                  title="B = 10 MHz", xlim=(0, 10), ylim=(-5, 0), \
                                  **plotting_styles["pop_style_b"])
    """
    # Default line styles for different populations (if no style provided)
    default_styles = [
        {'linewidth': 1.25, 'linestyle': '-'},
        {'linewidth': 1.25, 'linestyle': '--'},
        {'linewidth': 1.25, 'linestyle': '-.'},
        {'linewidth': 1.25, 'linestyle': ':'},
    ]
    
    # Use provided styles or default styles
    if plotting_style_list is None:
        plotting_style_list = default_styles
    
    # Plot each population curve
    for idx, (y_data, label) in enumerate(zip(pop_list, legend_list)):
        y = np.asarray(y_data)
        if log_scale:
            y = np.log10(np.maximum(y_data, 1e-12))  # Avoid log(0)
        
        # Get style for this curve
        style = plotting_style_list[idx % len(plotting_style_list)]
        
        ax.plot(tlist, y, label=label, **style)
    
    # Plot pulse shape if provided
    if plot_pulse and pulse_dict:
        for pulse_name, pulse_info in pulse_dict.items():
            # Get pulse data
            data = pulse_info['data']
            # If data is callable, evaluate it
            if callable(data):
                args = pulse_info.get('args', {})
                pulse_values = np.array([data(t, **args) for t in tlist])
            else:
                pulse_values = np.asarray(data)
            
            # Get plot style
            style_name = pulse_info.get('style', None)
            if style_name and style_name in plotting_styles:
                plot_style = plotting_styles[style_name]
            else:
                plot_style = {}
            
            # Get label
            label = pulse_info.get('label', pulse_name)
            
            # Plot
            ax.plot(tlist, pulse_values, label=label, **plot_style)
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        if log_scale:
            ax.set_ylabel(r'$\log_{10}$(Population)')
        else:
            ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim((min(tlist), max(tlist)))
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        if log_scale:
            ax.set_ylim((-12, 0.1))
        else:
            ax.set_ylim((-0.1, 1.1))
    
    # Add grid and legend
    if grid_alpha:
        ax.grid(True, alpha=grid_alpha)
    
    # Add legend
    if legend_list and show_legend:
        ax.legend(**legend_styles['outside_left'])
    return ax


def plot_pulse_shapes(
    ax: plt.Axes,
    tlist: Sequence[float],
    pulse_dict: dict,
    title: Optional[str] = None,
    xlabel: str = r"Time ($\mu$s)",
    ylabel: str = "Amplitude (MHz)",
    normalize_2pi: bool = True,
    ylim: Optional[tuple] = None,
    grid_alpha: float = 0.3,
    legend_config: Optional[dict] = legend_styles['compact'],
) -> plt.Axes:
    """
    Plot pulse shapes (e.g., Omega_r, Delta_r) on a single axes.
    
    Parameters:
    ----------
    ax : plt.Axes
        The axes to plot on
    tlist : Sequence[float]
        Time points
    pulse_dict : dict
        Dictionary with pulse data. Format:
        {
            'pulse_name': {
                'data': array_like or callable,
                'args': dict (optional, if data is callable),
                'label': str,
                'style': str (optional, key in plotting_styles),
                'normalize_factor': float (optional, overrides normalize_2pi for this pulse)
            }
        }
    title : str, optional
        Title for the axes
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    normalize_2pi : bool
        If True, divide pulse amplitudes by 2π for display
    ylim : tuple, optional
        Y-axis limits as (ymin, ymax)
    grid_alpha : float
        Grid transparency
    legend_loc : str
        Legend location
    legend_bbox : tuple, optional
        Legend bbox_to_anchor parameter
    **legend_kwargs : dict
        Additional keyword arguments for legend
    
    Returns:
    -------
    ax : plt.Axes
        The modified axes object
    
    Example:
    -------
    >>> # Method 1: Pass pre-computed pulse data
    >>> pulse_dict = {
    ...     'Omega_r': {
    ...         'data': Omega_r_pulse_shape / (2*np.pi),
    ...         'label': r'$\Omega_r$',
    ...         'style': 'Omega_r_pulse'
    ...     },
    ...     'Delta_r': {
    ...         'data': Delta_r_pulse_shape / (2*np.pi),
    ...         'label': r'$\Delta_r$',
    ...         'style': 'Delta_r_pulse'
    ...     }
    ... }
    >>> plot_pulse_shapes(ax, tlist, pulse_dict, ylim=(-25, 25))
    
    >>> # Method 2: Pass callable functions with args
    >>> pulse_dict = {
    ...     'Omega_r': {
    ...         'data': APR_pulse_shape_Omega_r,
    ...         'args': {'amp_Omega_r': 2*np.pi*17, 'T_gate': 0.54, 'tau': 0.09},
    ...         'label': r'$\Omega_r$',
    ...         'style': 'Omega_r_pulse'
    ...     }
    ... }
    """
    for pulse_name, pulse_info in pulse_dict.items():
        # Get pulse data
        data = pulse_info['data']
        
        # If data is callable, evaluate it
        if callable(data):
            args = pulse_info.get('args', {})
            pulse_values = np.array([data(t, **args) for t in tlist])
        else:
            pulse_values = np.asarray(data)
        
        # Normalize: use pulse-specific normalize_factor if provided, otherwise use global normalize_2pi
        if 'normalize_factor' in pulse_info:
            normalize_factor = pulse_info['normalize_factor']
            if normalize_factor != 1.0 and normalize_factor != 0:
                pulse_values = pulse_values / normalize_factor
        elif normalize_2pi:
            pulse_values = pulse_values / (2 * np.pi)
        
        # Get plot style
        style_name = pulse_info.get('style', None)
        if style_name and style_name in plotting_styles:
            plot_style = plotting_styles[style_name]
        else:
            plot_style = {}
        
        # Get label
        label = pulse_info.get('label', pulse_name)
        
        # Plot
        ax.plot(tlist, pulse_values, label=label, **plot_style)
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Set y-axis limits
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add grid
    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha)
    
    # Add legend
    if legend_config is None:
        legend_config = legend_styles['compact']
    ax.legend(**legend_config)
    
    return ax


def plot_multiple_population_evolution(
    fig: plt.Figure,
    axes: np.ndarray,
    tlist: Sequence[float],
    pop_dict_list: Sequence[dict],
    suptitle: str = "",
    legend_config: Optional[dict] = None,
    global_plot_pulse: bool = False,
    global_pulse_dict: Optional[dict] = None,
) -> plt.Figure:
    """
    Plot population evolution on multiple subplots by calling plot_population_evolution.
    
    Parameters:
    ----------
    fig : plt.Figure
        The figure object
    axes : np.ndarray
        Array of axes objects (should be flattened with .ravel())
    tlist : Sequence[float]
        Time points
    pop_dict_list : Sequence[dict]
        List of configuration dictionaries for population plots, one per subplot.
        Each dict can have:
        - 'result': QuTiP result object from mesolve()
        - 'indices': Sequence[int], indices to plot from result.expect
        - 'labels': Sequence[str], labels for each population curve
        - 'plot_pulse': bool (optional), whether to plot pulse overlay (overrides global_plot_pulse)
        - 'pulse_dict': dict (optional), pulse configuration (overrides global_pulse_dict)
        - 'sub_title': str (optional), subplot title
        - 'xlabel': str or None (optional), x-axis label. None = no label
        - 'ylabel': str or None (optional), y-axis label. None = no label
        - 'xlim': tuple (optional), x-axis limits
        - 'ylim': tuple (optional), y-axis limits
        - 'grid_alpha': float (optional), grid transparency
        - 'plotting_style_list': Sequence[dict] (optional), list of styles for each curve
        
        Example 1 - No pulse:
        pop_dict_list = [
            {
                'result': res, 
                'indices': [0, 1], 
                'labels': [r'$|1\\rangle$', r'$|r\\rangle$'],
                'sub_title': r'$\\Omega_r = 10$ MHz',
            }
            for res in results_list
        ]
        
        Example 2 - All subplots share the same pulse:
        global_pulse_dict = {
            'gaussian_pulse': {
                'data': scale_pulse_shape,
                'label': 'Gaussian',
                'style': 'Gaussian_pulse'
            }
        }
        plot_multiple_population_evolution(
            fig, axes, tlist, pop_dict_list,
            global_plot_pulse=True,
            global_pulse_dict=global_pulse_dict
        )
        
        Example 3 - Different pulses for different subplots:
        pop_dict_list = [
            {
                'result': res,
                'indices': [0, 1],
                'labels': [r'$|1\\rangle$', r'$|r\\rangle$'],
                'plot_pulse': True,
                'pulse_dict': {
                    'gaussian_pulse': {
                        'data': gaussian_pulse,
                        'args': {'pulse_width': sigma, 'pulse_center': t0, 'Amplitude': Omega},
                        'label': f'Ω={Omega} MHz',
                        'style': 'Gaussian_pulse'
                    }
                }
            }
            for res, Omega in zip(results_list, Omega_list)
        ]
        
    suptitle : str
        Main title for the entire figure
    legend_config : dict, optional
        Configuration dictionary for the figure legend.
        Use legend_styles['default'], legend_styles['compact'], etc.
        If None, uses legend_styles['default']
    global_plot_pulse : bool
        Whether to plot pulse overlay on all subplots (can be overridden by pop_dict)
    global_pulse_dict : dict, optional
        Pulse configuration to use for all subplots (can be overridden by pop_dict).
        Format same as pulse_dict in plot_population_evolution:
        {
            'pulse_name': {
                'data': array_like or callable,
                'args': dict (optional, if data is callable),
                'label': str,
                'style': str (optional, key in plotting_styles)
            }
        }
    
    Returns:
    -------
    fig : plt.Figure
        The modified figure object
    """
    # Plot on each subplot
    for ax, pop_dict in zip(axes, pop_dict_list):
        # Extract result and indices
        result = pop_dict['result']
        pop_indices = pop_dict['indices']
        pop_labels = pop_dict['labels']
        
        # Build pop_list from result
        pop_list = [result.expect[idx] for idx in pop_indices]
        
        # Extract optional parameters
        # Priority: pop_dict settings > global settings
        plot_pulse = pop_dict.get('plot_pulse', global_plot_pulse)
        pulse_dict = pop_dict.get('pulse_dict', global_pulse_dict if global_pulse_dict else {})
        title = pop_dict.get('sub_title', None)
        xlabel = pop_dict.get('xlabel', r"Time ($\mu$s)")
        ylabel = pop_dict.get('ylabel', "Population")
        log_scale = pop_dict.get('log_scale', False)
        xlim = pop_dict.get('xlim', None)
        ylim = pop_dict.get('ylim', None)
        grid_alpha = pop_dict.get('grid_alpha', 0.3)
        plotting_style_list = pop_dict.get('plotting_style_list', None)
        
        # Call plot_population_evolution (without showing individual legends)
        plot_population_evolution(
            ax=ax,
            tlist=tlist,
            pop_list=pop_list,
            legend_list=pop_labels,
            plot_pulse=plot_pulse,
            pulse_dict=pulse_dict,
            title=title,
            xlabel=xlabel if xlabel is not None else "",
            ylabel=ylabel if ylabel is not None else "",
            log_scale=log_scale,
            xlim=xlim,
            ylim=ylim,
            grid_alpha=grid_alpha,
            show_legend=False,  # Don't show legend on individual subplots
            plotting_style_list=plotting_style_list,
        )
    
    # Create figure legend from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Use default legend config if not provided
    if legend_config is None:
        legend_config = legend_styles['default']
    
    fig.legend(handles, labels, **legend_config)
    
    # Set main title
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    
    return fig


def plot_fidelity_vs_parameter(
    ax: plt.Axes,
    param_list: Sequence[float],
    fidelity_list: Sequence[np.ndarray],
    legend_list: Sequence[str],
    title: Optional[str] = None,
    xlabel: str = "Parameter",
    ylabel: str = "Fidelity",
    use_infidelity: bool = False,
    log_scale: bool = False,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    grid_alpha: float = 0.3,
    show_legend: bool = True,
    plotting_style_list: Optional[Sequence[dict]] = None,
) -> plt.Axes:
    """
    Plot fidelity vs parameter on a single axes.
    
    Parameters:
    ----------
    ax : plt.Axes
        The axes to plot on
    param_list : Sequence[float]
        Parameter values (x-axis)
    fidelity_list : Sequence[np.ndarray]
        List of fidelity data arrays to plot (one per curve)
    legend_list : Sequence[str]
        List of labels for each fidelity curve
    title : str, optional
        Title for the axes
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    use_infidelity : bool
        If True, plot 1-fidelity (infidelity) instead of fidelity
    log_scale : bool
        If True, use logarithmic scale for y-axis (commonly used with infidelity)
    xlim : tuple, optional
        X-axis limits as (xmin, xmax)
    ylim : tuple, optional
        Y-axis limits as (ymin, ymax)
    grid_alpha : float
        Grid transparency
    show_legend : bool
        Whether to show legend on this axes
    plotting_style_list : Sequence[dict], optional
        List of style dictionaries for each fidelity curve.
        If None, uses default styles
    
    Returns:
    -------
    ax : plt.Axes
        The modified axes object
    
    Example:
    -------
    >>> fig, ax = plt.subplots()
    >>> fidelity_list = [fidelity_17MHz, fidelity_8p5MHz, fidelity_34MHz]
    >>> legend_list = [r"17 MHz", r"8.5 MHz", r"$\Omega_{max}/2\pi = 34$ MHz"]
    >>> plotting_style_list = [
    ...     plotting_styles['fidelity_style_a'],
    ...     plotting_styles['fidelity_style_b'],
    ...     plotting_styles['fidelity_style_c']
    ... ]
    >>> # Plot infidelity on log scale
    >>> plot_fidelity_vs_parameter(ax, B_list, fidelity_list, legend_list,
    ...                            xlabel=r"$B/2\pi$ (MHz)", 
    ...                            ylabel=r"$1-\mathcal{F}$",
    ...                            use_infidelity=True,
    ...                            log_scale=True,
    ...                            plotting_style_list=plotting_style_list)
    >>> # Plot fidelity on linear scale
    >>> plot_fidelity_vs_parameter(ax, B_list, fidelity_list, legend_list,
    ...                            xlabel=r"$B/2\pi$ (MHz)",
    ...                            use_infidelity=False,
    ...                            log_scale=False,
    ...                            plotting_style_list=plotting_style_list)
    """
    # Default styles if none provided
    default_styles = [
        plotting_styles['fidelity_style_a'],
        plotting_styles['fidelity_style_b'],
        plotting_styles['fidelity_style_c'],
        plotting_styles['fidelity_style_d'],
    ]
    
    # Use provided styles or default styles
    if plotting_style_list is None:
        plotting_style_list = default_styles
    
    # Plot each fidelity curve
    for idx, (fidelity_data, label) in enumerate(zip(fidelity_list, legend_list)):
        y = np.asarray(fidelity_data)
        
        if use_infidelity:
            # Plot 1-F (infidelity)
            y = np.maximum(1 - y, 1e-12)  # Avoid negative values or zero for log scale
        
        # Get style for this curve
        style = plotting_style_list[idx % len(plotting_style_list)]
        
        ax.plot(param_list, y, label=label, **style)
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        if use_infidelity and ylabel == "Fidelity":
            ax.set_ylabel(r'$1-F$')
        else:
            ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim((min(param_list), max(param_list)))  
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        if use_infidelity:
            if log_scale:
                ax.set_ylim((1e-3, 1))
            else:
                ax.set_ylim((0, 1))
        else:
            ax.set_ylim((0, 1.05))
    
    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Add grid
    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha)
    
    # Add legend
    if show_legend:
        ax.legend(**legend_styles['compact'])
    
    return ax


def plot_optimization_convergence(history, param_names, figsize=(12, 8)):
    """
    Plot optimization convergence history.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing optimization history with keys:
        - 'iteration': list of iteration numbers
        - 'fidelity': list of fidelity values
        - 'infidelity': list of infidelity values
        - param_names[i]: list of parameter values for each parameter
    param_names : list
        Names of optimized parameters
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    
    Examples:
    ---------
    >>> history = {
    ...     'iteration': [1, 2, 3, ...],
    ...     'fidelity': [0.95, 0.97, 0.98, ...],
    ...     'infidelity': [0.05, 0.03, 0.02, ...],
    ...     'T_gate': [0.5, 0.52, 0.54, ...],
    ...     'tau': [0.1, 0.095, 0.09, ...],
    ...     ...
    ... }
    >>> param_names = ['T_gate', 'tau', 'amp_Omega_r', 'amp_Delta_r']
    >>> fig, axes = plot_optimization_convergence(history, param_names)
    >>> plt.show()
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    
    # Plot fidelity convergence
    ax = axes[0, 0]
    ax.plot(history['iteration'], history['fidelity'], 'o-', linewidth=2, markersize=4, markevery=5)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_title('Fidelity Convergence', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Plot infidelity (log scale)
    ax = axes[0, 1]
    ax.semilogy(history['iteration'], history['infidelity'], 'o-', 
                linewidth=2, markersize=4, markevery=5, color='red')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Infidelity (log scale)', fontsize=11)
    ax.set_title('Infidelity Convergence', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Plot parameter evolution
    ax = axes[1, 0]
    for name in param_names:
        if name in history:
            ax.plot(history['iteration'], history[name], 'o-', 
                   label=name, linewidth=2, markersize=4, markevery=5)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Parameter Value', fontsize=11)
    ax.set_title('Parameter Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Plot normalized parameter evolution
    ax = axes[1, 1]
    for name in param_names:
        if name in history:
            values = np.array(history[name])
            normalized = (values - values[0]) / (np.abs(values[0]) + 1e-10)
            ax.plot(history['iteration'], normalized, 'o-', 
                   label=name, linewidth=2, markersize=4, markevery=5)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Normalized Change', fontsize=11)
    ax.set_title('Normalized Parameter Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    
    return fig, axes


def plot_parameter_vs_parameter(
    ax: plt.Axes,
    x_param_list: Sequence[float],
    y_param_list: Sequence[np.ndarray],
    legend_list: Sequence[str],
    title: Optional[str] = None,
    xlabel: str = "Parameter",
    ylabel: str = "Value",
    log_scale_x: bool = False,
    log_scale_y: bool = False,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    grid_alpha: float = 0.3,
    show_legend: bool = True,
    plotting_style_list: Optional[Sequence[dict]] = None,
    reference_lines: Optional[dict] = None,
) -> plt.Axes:
    """
    Plot parameter vs parameter on a single axes (generalized version of plot_fidelity_vs_parameter).
    
    Parameters:
    ----------
    ax : plt.Axes
        The axes to plot on
    x_param_list : Sequence[float]
        X-axis parameter values
    y_param_list : Sequence[np.ndarray]
        List of Y-axis parameter arrays to plot (one per curve)
    legend_list : Sequence[str]
        List of labels for each curve
    title : str, optional
        Title for the axes
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    log_scale_x : bool
        If True, use logarithmic scale for x-axis
    log_scale_y : bool
        If True, use logarithmic scale for y-axis
    xlim : tuple, optional
        X-axis limits as (xmin, xmax)
    ylim : tuple, optional
        Y-axis limits as (ymin, ymax)
    grid_alpha : float
        Grid transparency
    show_legend : bool
        Whether to show legend
    plotting_style_list : Sequence[dict], optional
        List of style dictionaries for each curve
    reference_lines : dict, optional
        Dictionary of reference lines to add. Format:
        {
            'horizontal': [(y_value, color, linestyle, label), ...],
            'vertical': [(x_value, color, linestyle, label), ...]
        }
    
    Returns:
    -------
    ax : plt.Axes
        The modified axes object
    
    Example:
    -------
    >>> fig, ax = plt.subplots()
    >>> # Plot fidelity vs B
    >>> plot_parameter_vs_parameter(
    ...     ax, scale_B_values, [fidelities],
    ...     legend_list=['Fidelity'],
    ...     xlabel=r'$B/2\pi$ (MHz)', ylabel='Fidelity',
    ...     reference_lines={'horizontal': [(0.99, 'red', '--', 'Target')]}
    ... )
    >>> # Plot T_gate vs B
    >>> plot_parameter_vs_parameter(
    ...     ax, scale_B_values, [T_gate_values],
    ...     legend_list=[r'$T_{gate}$'],
    ...     xlabel=r'$B/2\pi$ (MHz)', ylabel=r'$T_{gate}$ ($\mu$s)'
    ... )
    """
    # Default styles if none provided
    default_styles = [
        plotting_styles['fidelity_style_a'],
        plotting_styles['fidelity_style_b'],
        plotting_styles['fidelity_style_c'],
        plotting_styles['fidelity_style_d'],
    ]
    
    # Use provided styles or default styles
    if plotting_style_list is None:
        plotting_style_list = default_styles
    
    # Plot each curve
    for idx, (y_data, label) in enumerate(zip(y_param_list, legend_list)):
        y = np.asarray(y_data)
        
        # Get style for this curve
        style = plotting_style_list[idx % len(plotting_style_list)]
        
        ax.plot(x_param_list, y, label=label, **style)
    
    # Add reference lines if provided
    if reference_lines:
        if 'horizontal' in reference_lines:
            for line_info in reference_lines['horizontal']:
                y_val, color, linestyle, line_label = line_info
                ax.axhline(y=y_val, color=color, linestyle=linestyle, 
                          linewidth=1.5, alpha=0.7, label=line_label)
        
        if 'vertical' in reference_lines:
            for line_info in reference_lines['vertical']:
                x_val, color, linestyle, line_label = line_info
                ax.axvline(x=x_val, color=color, linestyle=linestyle,
                          linewidth=1.5, alpha=0.7, label=line_label)
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim((min(x_param_list), max(x_param_list)))
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Set log scale if requested
    if log_scale_x:
        ax.set_xscale('log')
    if log_scale_y:
        ax.set_yscale('log')
    
    # Add grid
    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha)
    
    # Add legend
    if show_legend:
        ax.legend(**legend_styles['compact'])
    
    return ax
