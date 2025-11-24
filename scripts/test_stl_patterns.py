import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


def test_pattern(name, data):
    print(f"Testing pattern: {name}")
    # Create a time series with hourly frequency
    series = pd.Series(data,
                       index=pd.date_range("2024-01-01",
                                           periods=len(data),
                                           freq="H"))

    # Perform STL decomposition
    # period=24 for daily seasonality in hourly data
    stl = STL(series, period=24, robust=True)
    res = stl.fit()

    # Plotting
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
    res.observed.plot(ax=axes[0], title=f"{name} - Observed")
    axes[0].set_ylabel("Observed")

    res.trend.plot(ax=axes[1], title="Trend")
    axes[1].set_ylabel("Trend")

    res.seasonal.plot(ax=axes[2], title="Seasonal")
    axes[2].set_ylabel("Seasonal")

    res.resid.plot(ax=axes[3], title="Residual")
    axes[3].set_ylabel("Residual")

    plt.tight_layout()
    output_file = f"stl_test_{name.replace(' ', '_').lower()}.png"
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    # plt.show() # Commented out for non-interactive environments


if __name__ == "__main__":
    # 1. Exponential Growth
    # y = e^(0.5x) + Seasonality
    x = np.linspace(0, 5, 500)
    seasonality = np.sin(x * 20) * 5  # High frequency seasonality
    exp_data = np.exp(x) * 10 + seasonality
    test_pattern("Exponential Growth", exp_data)

    # 2. Step Function
    # y = 10 for t < 250, y = 20 for t >= 250
    step_data = np.concatenate([np.ones(250) * 10, np.ones(250) * 20])
    # Add some noise and seasonality
    step_data += np.random.normal(0, 0.5, 500)
    step_data += np.sin(np.linspace(0, 50, 500)) * 2
    test_pattern("Step Function", step_data)

    # 3. Linear Trend
    linear_data = np.linspace(10, 50,
                              500) + np.sin(np.linspace(0, 50, 500)) * 5
    test_pattern("Linear Trend", linear_data)
