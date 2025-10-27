"""Anode lifetime simulation model.

This module generates synthetic anode lifetime data. The data is modeled to be
constant for a specified duration, then decay exponentially, with added noise.
"""

import numpy as np
import pandas as pd


def simulate_anode_lifetime(
    constant_duration: int,
    decay_rate: float,
    noise_level: float,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame:
    """
    Generates anode lifetime data.

    The model is:
    - Constant value of 1 for a certain duration.
    - Exponential decay after that duration.
    - Added Gaussian noise.

    Args:
        constant_duration: The number of hours the lifetime remains constant.
        decay_rate: The hourly rate of exponential decay.
        noise_level: The standard deviation of the Gaussian noise to add.
        start_date: The start date for the simulation (e.g., "2020-01-01").
        end_date: The end date for the simulation (e.g., "2022-12-31").
        interval: The data generation interval (e.g., '1h', '1d').

    Returns:
        A pandas DataFrame with a datetime index and an 'anode_lifetime' column.
    """
    datetime_range = pd.date_range(start=start_date, end=end_date, freq=interval)
    n_samples = len(datetime_range)
    anode_lifetime = np.ones(n_samples)

    # Find the index where decay should start
    if n_samples > 0:
        decay_start_time = datetime_range[0] + pd.to_timedelta(
            constant_duration, unit="h"
        )
        decay_start_index = datetime_range.searchsorted(decay_start_time)

        if decay_start_index < n_samples:
            # Scale the decay rate based on the interval to keep it consistent
            interval_in_hours = pd.to_timedelta(interval).total_seconds() / 3600
            scaled_decay_rate = decay_rate * interval_in_hours

            # Generate decay values
            decay_steps = n_samples - decay_start_index
            time_decay = np.arange(decay_steps)
            decay_values = np.exp(-scaled_decay_rate * time_decay)
            anode_lifetime[decay_start_index:] = decay_values

    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    anode_lifetime_with_noise = anode_lifetime + noise

    # Clip values to a realistic range [0, 1.1]
    anode_lifetime_clipped = np.clip(anode_lifetime_with_noise, 0, 1.1)

    df = pd.DataFrame(
        anode_lifetime_clipped, index=datetime_range, columns=["anode_lifetime"]
    )
    return df