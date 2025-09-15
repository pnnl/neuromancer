"""
plot.py

Standalone plotting function that works with both BuildingComponent and BuildingSystem objects.
"""

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from typing import Dict, List, Tuple, Optional, Union, Any


def simplot(
        model: Union['BuildingComponent', 'BuildingSystem'],
        results: Optional[Dict[str, torch.Tensor]] = None,
        # Plotting-specific parameters
        variables: Optional[List[str]] = None,  # List of variables to plot, None = all
        time_range: Optional[Tuple[float, float]] = None,  # (start_hour, end_hour) for zooming
        filename: Optional[str] = None,  # Save figure to this file
        figsize: Optional[Tuple[float, float]] = None,  # Figure size (width, height)
        title: Optional[str] = None,  # Custom title
        batch_idx: int = 0,  # Which batch element to plot

        # Simulation parameters (if results not provided)
        **kwargs
) -> Tuple[plt.Figure, Dict[str, torch.Tensor]]:
    """
    Universal plotting function for BuildingComponent and BuildingSystem objects.

    With standardized interfaces (both use .simulate() and return [batch_size, time, dim]),
    no type checking is needed!

    Args:
        model: BuildingComponent or BuildingSystem instance to plot
        results (dict, optional): Pre-computed simulation results. If None, runs model.simulate().
        variables (list, optional): List of variable names to plot. If None, plots all.
        time_range (tuple, optional): (start_hour, end_hour) to zoom into specific period.
        filename (str, optional): Save figure to this file.
        figsize (tuple, optional): Figure size (width, height). Auto-calculated if None.
        title (str, optional): Custom title. Auto-generated if None.
        batch_idx (int): Which batch element to plot (default: 0).

        **kwargs: Simulation parameters passed to model.simulate().

    Returns:
        tuple: (fig, results) - matplotlib figure and simulation results dict

    Example usage:
        # Works identically for both types!
        fig, results = plot_building_simulation(envelope, duration_hours=24)
        fig, results = plot_building_simulation(system, duration_hours=24)
    """
    model_name = getattr(model, 'name', model.__class__.__name__)

    if results is None:
        # Standard simulation parameters
        sim_kwargs = {
            'duration_hours': 24.0,
            'dt_minutes': 5.0,
            't_start_hour': 5.0,  # Can be standardized name
            **kwargs
        }

        # Both models use same method and return same format!
        results = model.simulate(**sim_kwargs)

        duration_hours = sim_kwargs['duration_hours']
        dt_minutes = sim_kwargs['dt_minutes']
        t_start_hour = sim_kwargs.get('t_start_hour', 5.0)
    else:
        # Extract time parameters from existing results
        t_start_hour, dt_minutes, duration_hours = _extract_time_params_from_results(results, batch_idx)

    # Get time vector for plotting - same for both model types
    time_hours = _create_time_vector(results, batch_idx, t_start_hour, dt_minutes)

    # Select and filter variables
    var_names = _select_variables(results, variables)

    # Apply time range filtering - same logic for both
    time_hours_plot, results_plot = _apply_time_range_filter(
        results, var_names, time_hours, time_range, t_start_hour, batch_idx
    )

    # Create the plot
    fig = _create_plot(
        results_plot, var_names, time_hours_plot, time_range, t_start_hour,
        duration_hours, dt_minutes, batch_idx, model_name, figsize, title, filename
    )

    return fig, results


def _extract_time_params_from_results(results: Dict[str, torch.Tensor],
                                      batch_idx: int) -> Tuple[float, float, float]:
    """Extract time parameters from simulation results - same format for both model types."""
    if 't' in results:
        # Both return [batch_size, time] format
        time_tensor = results['t'][batch_idx, :]

        if len(time_tensor) > 1:
            dt_minutes = float(time_tensor[1] - time_tensor[0]) / 60.0
            t_start_hour = float(time_tensor[0]) / 3600.0
            duration_hours = float(time_tensor[-1] - time_tensor[0]) / 3600.0
        else:
            dt_minutes, t_start_hour, duration_hours = 5.0, 0.0, 24.0
    else:
        # Fallback defaults
        dt_minutes, t_start_hour, duration_hours = 5.0, 0.0, 24.0

    return t_start_hour, dt_minutes, duration_hours


def _create_time_vector(results: Dict[str, torch.Tensor],
                        batch_idx: int,
                        t_start_hour: float,
                        dt_minutes: float) -> torch.Tensor:
    """Create time vector for plotting - same format for both model types."""
    dt = dt_minutes * 60.0
    t_start = t_start_hour * 3600

    # Both return [batch_size, time, ...] format
    # Find the maximum time dimension across all variables (excluding 't' and 'dt')
    data_vars = [k for k in results.keys() if k not in ['t', 'dt']]
    if data_vars:
        sample_result = results[data_vars[0]]
        n_steps = sample_result.shape[1]  # time is always dimension 1
    else:
        # Fallback to 't' if no other variables
        sample_result = next(iter(results.values()))
        n_steps = sample_result.shape[1]

    time_points = torch.arange(n_steps) * dt + t_start
    return time_points.cpu().numpy() / 3600.0


def _select_variables(results: Dict[str, torch.Tensor],
                      variables: Optional[List[str]]) -> List[str]:
    """Select which variables to plot."""
    if variables is None:
        # Plot all variables except time parameters
        var_names = [k for k in results.keys() if k not in ['t', 'dt']]
    else:
        var_names = []
        for var in variables:
            if var in results:
                var_names.append(var)
            else:
                print(f"Warning: Variable '{var}' not found in results. Available: {list(results.keys())}")

    if not var_names:
        raise ValueError("No valid variables to plot")

    return var_names


def _apply_time_range_filter(results: Dict[str, torch.Tensor],
                             var_names: List[str],
                             time_hours: torch.Tensor,
                             time_range: Optional[Tuple[float, float]],
                             t_start_hour: float,
                             batch_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply time range filtering to results - same format for both model types."""
    results_plot = {}

    # Process each variable and ensure time alignment
    for var in var_names:
        data = results[var][batch_idx, :].detach().cpu().numpy()

        # Handle potential mismatch between time and data lengths
        if len(data) != len(time_hours):
            if len(data) == len(time_hours) + 1:
                # Data has one more point than time (initial condition + n steps)
                # Create extended time array
                dt_hours = time_hours[1] - time_hours[0] if len(time_hours) > 1 else 1/12  # 5min default
                extended_time = torch.cat([
                    torch.tensor([time_hours[0] - dt_hours]),
                    torch.tensor(time_hours)
                ])
                time_hours_for_var = extended_time
            elif len(data) == len(time_hours) - 1:
                # Data has one fewer point than time
                time_hours_for_var = time_hours[:-1]
            else:
                # Other mismatches - truncate to shorter length
                min_len = min(len(data), len(time_hours))
                data = data[:min_len]
                time_hours_for_var = time_hours[:min_len]
        else:
            time_hours_for_var = time_hours

        # Store the aligned time for this variable
        if var == var_names[0]:  # Use first variable's time as reference
            time_hours_plot = time_hours_for_var

        results_plot[var] = data

    # Apply time range filtering if specified
    if time_range is not None:
        start_hour, end_hour = time_range
        start_abs = t_start_hour + start_hour
        end_abs = t_start_hour + end_hour

        # Find indices for time range
        mask = (time_hours_plot >= start_abs) & (time_hours_plot <= end_abs)
        time_hours_plot = time_hours_plot[mask]

        # Filter all results with the same mask
        for var in var_names:
            if len(results_plot[var]) == len(mask):
                results_plot[var] = results_plot[var][mask]
            else:
                # Handle length mismatch in masking
                var_mask = mask[:len(results_plot[var])]
                results_plot[var] = results_plot[var][var_mask]

    return time_hours_plot, results_plot


def _create_plot(results_plot: Dict[str, torch.Tensor],
                 var_names: List[str],
                 time_hours_plot: torch.Tensor,
                 time_range: Optional[Tuple[float, float]],
                 t_start_hour: float,
                 duration_hours: float,
                 dt_minutes: float,
                 batch_idx: int,
                 model_name: str,
                 figsize: Optional[Tuple[float, float]],
                 title: Optional[str],
                 filename: Optional[str]) -> plt.Figure:
    """Create the actual matplotlib plot."""
    # Calculate figure size
    n_vars = len(var_names)
    if figsize is None:
        width = 12 if time_range else 8
        height = min(2.5 * n_vars, 20)  # Cap at reasonable height
        figsize = (width, height)

    # Create plots
    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    if n_vars == 1:
        axes = [axes]

    for i, var in enumerate(var_names):
        data = results_plot[var]

        # Handle different tensor dimensions
        if data.ndim >= 2:
            # For multi-dimensional outputs, plot first dimension
            if data.shape[1] == 1:  # Shape like [time, 1]
                y_data = data[:, 0]
            elif data.ndim == 2:  # Shape like [time, features]
                y_data = data[:, 0]  # Plot first feature
            else:  # Shape like [time, zone1, zone2, ...]
                y_data = data[:, 0]  # Plot first zone/feature
        else:  # 1D data
            y_data = data

        # Ensure time and data arrays have compatible lengths
        time_plot = time_hours_plot[:len(y_data)]

        axes[i].plot(time_plot, y_data, label=var, linewidth=1.5)
        axes[i].set_ylabel(var)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='best')

        # Add value annotations for zoomed plots
        if time_range and len(time_plot) < 100:
            # Annotate start and end values for short time series
            axes[i].annotate(f'{y_data[0]:.3f}',
                             xy=(time_plot[0], y_data[0]),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.7)
            axes[i].annotate(f'{y_data[-1]:.3f}',
                             xy=(time_plot[-1], y_data[-1]),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.7)

    # Apply time formatting
    time_formatter = _create_time_formatter(t_start_hour, time_range)

    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

        if time_range and (time_range[1] - time_range[0]) <= 4:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Set xlabel
    if time_range and (time_range[1] - time_range[0]) <= 4:
        axes[-1].set_xlabel("Time")
    else:
        start_hour_12 = t_start_hour % 12
        if start_hour_12 == 0:
            start_hour_12 = 12
        start_am_pm = "AM" if t_start_hour < 12 else "PM"
        axes[-1].set_xlabel(f"Time (Starting {int(start_hour_12)} {start_am_pm})")

    # Set title
    if title is None:
        if time_range:
            title = (f"{model_name} Simulation: "
                     f"{time_range[1] - time_range[0]:.1f} hours "
                     f"(dt={dt_minutes:.1f}min, batch={batch_idx})")
        else:
            start_hour_12 = t_start_hour % 12
            if start_hour_12 == 0:
                start_hour_12 = 12
            start_am_pm = "AM" if t_start_hour < 12 else "PM"
            title = (f"{model_name} Simulation: "
                     f"{duration_hours:.1f} hours from {int(start_hour_12)} {start_am_pm} "
                     f"(dt={dt_minutes:.1f}min, batch={batch_idx})")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches='tight')

    return fig


def _create_time_formatter(t_start_hour: float,
                           time_range: Optional[Tuple[float, float]] = None):
    """Create time formatter for matplotlib."""

    def format_time_tick(x, pos):
        hour_24 = (t_start_hour + x) % 24

        if time_range and (time_range[1] - time_range[0]) <= 4:
            h = int(hour_24)
            m = int((hour_24 % 1) * 60)
            am_pm = "AM" if h < 12 else "PM"
            h_12 = h % 12
            if h_12 == 0:
                h_12 = 12
            return f"{h_12}:{m:02d} {am_pm}"
        else:
            hour_12 = hour_24 % 12
            if hour_12 == 0:
                hour_12 = 12
            am_pm = "AM" if hour_24 < 12 else "PM"
            return f"{int(hour_12)} {am_pm}"

    return format_time_tick