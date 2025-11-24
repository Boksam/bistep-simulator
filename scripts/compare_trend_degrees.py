import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error


def generate_pattern(pattern_type, length=200):
    x = np.linspace(0, 10, length)
    if pattern_type == "exponential_growth":
        # y = e^x
        y = np.exp(x * 0.5) * 10 + np.random.normal(0, 2, length)
    elif pattern_type == "exponential_decay":
        # y = e^-x
        y = np.exp(-x * 0.5) * 100 + np.random.normal(0, 2, length)
    elif pattern_type == "step_function":
        # Sigmoid-like step
        y = 100 / (1 + np.exp(-5 * (x - 5))) + np.random.normal(0, 2, length)

    # Add some seasonality to make it realistic for STL
    seasonality = np.sin(x * 10) * 5
    return pd.Series(y + seasonality,
                     index=pd.date_range(start="2024-01-01",
                                         periods=length,
                                         freq="h"))


def analyze_trend_degrees(pattern_name, series):
    print(f"\n--- Analyzing {pattern_name} ---")

    # 1. STL Decomposition (The "Truth" for the trend)
    stl = STL(series, period=24, robust=True)
    res = stl.fit()
    stl_trend = res.trend

    # Prepare X for fitting (numeric)
    x_numeric = np.arange(len(series))

    # Prepare X for plotting (including FUTURE extrapolation)
    # We will predict 20% further into the future to show stability
    future_steps = int(len(series) * 0.2)
    x_future = np.arange(len(series) + future_steps)

    # 2. Fit Polynomials (Degree 1, 2, and High Degree 15)
    # Degree 1 (Linear)
    coeffs_1 = np.polyfit(x_numeric, stl_trend, 1)
    model_1 = np.poly1d(coeffs_1)
    trend_1 = model_1(x_future)

    # Degree 2 (Quadratic)
    coeffs_2 = np.polyfit(x_numeric, stl_trend, 2)
    model_2 = np.poly1d(coeffs_2)
    trend_2 = model_2(x_future)

    # Degree 15 (High Complexity)
    coeffs_15 = np.polyfit(x_numeric, stl_trend, 15)
    model_15 = np.poly1d(coeffs_15)
    trend_15 = model_15(x_future)

    # 3. Calculate Errors (RMSE vs STL Trend on TRAINING data only)
    rmse_1 = np.sqrt(mean_squared_error(stl_trend, trend_1[:len(series)]))
    rmse_2 = np.sqrt(mean_squared_error(stl_trend, trend_2[:len(series)]))
    rmse_15 = np.sqrt(mean_squared_error(stl_trend, trend_15[:len(series)]))

    print(f"{'Degree':<15} | {'RMSE (Fit)':<12} | {'Description'}")
    print("-" * 50)
    print(f"{'1 (Linear)':<15} | {rmse_1:.4f}       | Stable but underfits")
    print(f"{'2 (Quad)':<15} | {rmse_2:.4f}       | Good balance")
    print(f"{'15 (High)':<15} | {rmse_15:.4f}       | Overfits & Unstable")

    # 4. Plot
    plt.figure(figsize=(12, 6))

    # Plot original Data
    plt.plot(np.arange(len(series)),
             series,
             label="Original Data (Noisy)",
             color="gray",
             alpha=0.3)

    # Plot original STL Trend
    plt.plot(np.arange(len(series)),
             stl_trend,
             label="STL Extracted Trend (Target)",
             color="black",
             linewidth=3,
             alpha=0.5)

    # Plot Fits
    plt.plot(x_future,
             trend_1,
             label=f"Degree 1 (RMSE={rmse_1:.1f})",
             linestyle="--")
    plt.plot(x_future,
             trend_2,
             label=f"Degree 2 (RMSE={rmse_2:.1f})",
             linestyle="--")
    plt.plot(x_future,
             trend_15,
             label=f"Degree 15 (RMSE={rmse_15:.1f})",
             linestyle="-",
             color="red")

    # Mark the "Future" zone
    plt.axvline(x=len(series),
                color='gray',
                linestyle=':',
                label="Start of Simulation (Future)")
    plt.axvspan(len(series), len(x_future), color='gray', alpha=0.1)

    plt.title(f"Trend Fitting: {pattern_name} (Extrapolation Test)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set Y-limits based on data range to avoid high-degree polynomial explosion affecting the view
    data_min, data_max = series.min(), series.max()
    data_range = data_max - data_min
    plt.ylim(data_min - data_range * 0.5, data_max + data_range * 0.5)

    filename = f"trend_compare_high_degree_{pattern_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


# Run for all patterns
patterns = ["Exponential Growth", "Exponential Decay", "Step Function"]
for p in patterns:
    s = generate_pattern(p.lower().replace(" ", "_"))
    analyze_trend_degrees(p, s)
