import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit


def exponential_model(x, a, b, c):
    return a * np.exp(np.clip(b * x, -100, 100)) + c


def sigmoid_model(x, L, x0, k, b):
    return L / (1 + np.exp(-np.clip(k * (x - x0), -100, 100))) + b


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

    # Fit Exponential
    try:
        # Initial guess: a=1, b=0.01 (slow growth), c=min
        p0_exp = [1, 0.01, np.min(stl_trend)]
        popt_exp, _ = curve_fit(exponential_model,
                                x_numeric,
                                stl_trend,
                                p0=p0_exp,
                                maxfev=10000)
        trend_exp = exponential_model(x_future, *popt_exp)
        rmse_exp = np.sqrt(
            mean_squared_error(stl_trend, trend_exp[:len(series)]))
    except Exception as e:
        print(f"Exponential fit failed: {e}")
        trend_exp = None
        rmse_exp = np.nan

    # Fit Sigmoid
    try:
        # Initial guess: L=range, x0=mid, k=0.1, b=min
        p0_sig = [
            np.max(stl_trend) - np.min(stl_trend),
            np.median(x_numeric), 0.1,
            np.min(stl_trend)
        ]
        popt_sig, _ = curve_fit(sigmoid_model,
                                x_numeric,
                                stl_trend,
                                p0=p0_sig,
                                maxfev=10000)
        trend_sig = sigmoid_model(x_future, *popt_sig)
        rmse_sig = np.sqrt(
            mean_squared_error(stl_trend, trend_sig[:len(series)]))
    except Exception as e:
        print(f"Sigmoid fit failed: {e}")
        trend_sig = None
        rmse_sig = np.nan

    # 3. Calculate Errors (RMSE vs STL Trend on TRAINING data only)
    rmse_1 = np.sqrt(mean_squared_error(stl_trend, trend_1[:len(series)]))
    rmse_2 = np.sqrt(mean_squared_error(stl_trend, trend_2[:len(series)]))
    rmse_15 = np.sqrt(mean_squared_error(stl_trend, trend_15[:len(series)]))

    print(f"{'Model':<15} | {'RMSE (Fit)':<12} | {'Description'}")
    print("-" * 50)
    print(f"{'Poly Deg 1':<15} | {rmse_1:.4f}       | Linear")
    print(f"{'Poly Deg 2':<15} | {rmse_2:.4f}       | Quadratic")
    print(f"{'Poly Deg 15':<15} | {rmse_15:.4f}       | High Degree Poly")
    if trend_exp is not None:
        print(
            f"{'Exponential':<15} | {rmse_exp:.4f}       | Exponential Growth/Decay"
        )
    if trend_sig is not None:
        print(f"{'Sigmoid':<15} | {rmse_sig:.4f}       | Logistic/S-curve")

    # 4. Plot
    plt.figure(figsize=(14, 8))  # Increased figure size for better visibility

    # Plot original Data
    plt.plot(np.arange(len(series)),
             series,
             label="Original Data",
             color="steelblue",
             alpha=0.6,
             linewidth=3)

    # Plot original STL Trend
    plt.plot(np.arange(len(series)),
             stl_trend,
             label="STL Trend (Target)",
             color="black",
             linewidth=1.5,
             alpha=0.8,
             zorder=5)  # Bring to front

    # Plot Fits with distinct styles
    plt.plot(x_future,
             trend_1,
             label=f"Poly Deg 1 (RMSE={rmse_1:.1f})",
             linestyle="--",
             color="navy",
             linewidth=1.5,
             alpha=0.8)

    plt.plot(x_future,
             trend_2,
             label=f"Poly Deg 2 (RMSE={rmse_2:.1f})",
             linestyle="--",
             color="darkorange",
             linewidth=1.5,
             alpha=0.8)

    plt.plot(x_future,
             trend_15,
             label=f"Poly Deg 15 (RMSE={rmse_15:.1f})",
             linestyle="--",
             color="red",
             linewidth=1.5,
             alpha=0.8)

    if trend_exp is not None:
        plt.plot(x_future,
                 trend_exp,
                 label=f"Exponential (RMSE={rmse_exp:.1f})",
                 linestyle="--",
                 color="forestgreen",
                 linewidth=1.5,
                 alpha=0.8)

    if trend_sig is not None:
        plt.plot(x_future,
                 trend_sig,
                 label=f"Sigmoid (RMSE={rmse_sig:.1f})",
                 linestyle="--",
                 color="magenta",
                 linewidth=1.5,
                 alpha=0.8)  # Mark the "Future" zone
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
