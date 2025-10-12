"""Plotting utilities for visualizing time series data.

This module provides functions to plot time series decomposition results and
to compare original vs. simulated data.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa import seasonal


def plot_decomposition(
    result: seasonal.DecomposeResult,
    sensor_type: str,
    output_dir: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
):
    """Plots the results of a time series decomposition.

    Can either save the plot to a file or draw on a provided figure object.

    Args:
        result: A DecomposeResult object from statsmodels.
        sensor_type: The type of sensor data being plotted.
        output_dir: Optional. The directory to save the plot image.
        fig: Optional. A matplotlib Figure to draw on.
    """
    if fig is None:
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    else:
        axes = fig.axes

    # Map sensor type to display names and units
    display_names = {"water_temperature": "Water Temperature", "salinity": "Salinity"}
    units = {"water_temperature": "°C", "salinity": "PSU"}
    display_name = display_names.get(sensor_type, "Value")
    unit = units.get(sensor_type, "")

    fig.suptitle(f"{display_name} Time Series Decomposition", fontsize=18)

    components = {
        "Observed": result.observed,
        "Trend": result.trend,
        "Seasonal": result.seasonal,
        "Residual": result.resid,
    }
    colors = ["cornflowerblue", "tomato", "forestgreen", "gray"]

    for (name, data), color, ax in zip(components.items(), colors, axes):
        plot_kwargs = {"color": color, "legend": False}
        if name == "Residual":
            plot_kwargs.update({"linestyle": "None", "marker": "."})

        data.plot(ax=ax, **plot_kwargs)

        y_label = name
        if name == "Observed":
            y_label = f"{name} ({unit})"

        ax.set_ylabel(y_label)
        ax.set_title(f"{name} Component", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("Date")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "time_series_decomposition.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Decomposition plot saved to: {plot_path}")
        plt.close(fig)


def plot_comparison(
    original: pd.Series,
    simulated: pd.Series,
    sensor_type: str,
    output_dir: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
):
    """Compares statistical properties of original and simulated data.

    This is especially useful when the simulated series is for a future period.

    Args:
        original: The original (historical) time series data.
        simulated: The simulated time series data.
        sensor_type: The type of sensor data being plotted.
        output_dir: Optional. The directory to save the plot image.
        fig: Optional. A matplotlib Figure to draw on.
    """
    if fig is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    else:
        axes = fig.axes

    # Map sensor type to display names and units
    display_names = {"water_temperature": "Water Temperature", "salinity": "Salinity"}
    units = {"water_temperature": "°C", "salinity": "PSU"}
    display_name = display_names.get(sensor_type, "Value")
    unit = units.get(sensor_type, "")

    # --- Filter historical data to match the simulation's seasonal range ---
    simulated_doy = simulated.index.dayofyear
    start_doy = simulated_doy.min()
    end_doy = simulated_doy.max()

    historical_doy = original.index.dayofyear
    if start_doy <= end_doy:
        is_in_range = (historical_doy >= start_doy) & (historical_doy <= end_doy)
    else:  # Handles year-spanning ranges (e.g., Dec to Jan)
        is_in_range = (historical_doy >= start_doy) | (historical_doy <= end_doy)

    filtered_original = original[is_in_range]

    # --- 1. Distribution Comparison (Histogram) ---
    ax1 = axes[0]
    filtered_original.plot(
        kind="hist",
        bins=50,
        ax=ax1,
        alpha=0.7,
        label="Historical (Matching Season)",
        color="cornflowerblue",
        density=True,
    )
    simulated.plot(
        kind="hist",
        bins=50,
        ax=ax1,
        alpha=0.7,
        label="Simulated",
        color="tomato",
        density=True,
    )
    ax1.set_title("Distribution Comparison", fontsize=14)
    ax1.set_xlabel(f"{display_name} ({unit})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 2. Seasonal Profile Comparison (by Day of Year) ---
    ax2 = axes[1]
    historical_seasonal = filtered_original.groupby(
        filtered_original.index.dayofyear
    ).agg(["mean", "std"])
    simulated_seasonal = simulated.groupby(simulated.index.dayofyear).mean()

    ax2.plot(
        historical_seasonal.index,
        historical_seasonal["mean"],
        label="Historical Avg.",
        color="cornflowerblue",
    )
    ax2.fill_between(
        historical_seasonal.index,
        historical_seasonal["mean"] - historical_seasonal["std"],
        historical_seasonal["mean"] + historical_seasonal["std"],
        color="cornflowerblue",
        alpha=0.2,
        label="Historical Std. Dev.",
    )
    ax2.plot(
        simulated_seasonal.index,
        simulated_seasonal,
        label="Simulated Avg.",
        color="tomato",
        marker="o",
        linestyle="--",
    )
    ax2.set_title("Seasonal Profile Comparison (by Day of Year)", fontsize=14)
    ax2.set_xlabel("Day of Year")
    ax2.set_ylabel(f"Avg. {display_name} ({unit})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- 3. Daily Profile Comparison (by Hour of Day) ---
    ax3 = axes[2]
    historical_daily = filtered_original.groupby(filtered_original.index.hour).agg(
        ["mean", "std"]
    )
    simulated_daily = simulated.groupby(simulated.index.hour).mean()

    ax3.plot(
        historical_daily.index,
        historical_daily["mean"],
        label="Historical Avg.",
        color="cornflowerblue",
    )
    ax3.fill_between(
        historical_daily.index,
        historical_daily["mean"] - historical_daily["std"],
        historical_daily["mean"] + historical_daily["std"],
        color="cornflowerblue",
        alpha=0.2,
        label="Historical Std. Dev.",
    )
    ax3.plot(
        simulated_daily.index,
        simulated_daily,
        label="Simulated Avg.",
        color="tomato",
        marker="o",
        linestyle="--",
    )
    ax3.set_title("Daily Profile Comparison (by Hour of Day)", fontsize=14)
    ax3.set_xlabel("Hour of Day (0-23)")
    ax3.set_ylabel(f"Avg. {display_name} ({unit})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(pad=3.0)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "comparison_analysis.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Comparison plot saved to: {plot_path}")
        plt.close(fig)
