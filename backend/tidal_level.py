"""Tidal level simulation model.

This module contains the TidalLevelSimulator class, which is responsible
for analyzing historical tidal level data and generating future synthetic
data based on the analysis.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa import ar_model, seasonal


class TidalLevelSimulator:
    """Simulates future tidal level data based on historical patterns.

    This class decomposes a time series of tidal level into trend,
    yearly seasonality, daily seasonality, and noise components. It models each
    component and then uses these models to generate synthetic data.

    The noise (residuals) is modeled using an Autoregressive (AR) model to
    capture realistic fluctuations.
    """

    def __init__(self):
        """Initializes the simulator."""
        self.trend_model: Optional[np.poly1d] = None
        self.yearly_seasonal_pattern: Optional[Dict[int, float]] = None
        self.daily_seasonal_pattern: Optional[np.ndarray] = None
        self.noise_model: Optional[Dict] = None
        self.original_data: Optional[pd.Series] = None
        self.full_decomposition_result: Optional[seasonal.DecomposeResult] = None

    def _handle_outliers(
        self,
        series: pd.Series,
        window: int = 24 * 7,
        sigma: float = 2.5,
    ) -> pd.Series:
        """Detects and removes outliers from the time series.

        Outliers are detected using a rolling window: values outside a `sigma`
        standard deviation range from the rolling median are considered outliers.

        Args:
            series: The input time series data.
            window: The size of the rolling window.
            sigma: The number of standard deviations for outlier detection.

        Returns:
            A new series with outliers replaced by NaN.
        """
        print("Detecting and handling outliers...")
        series_no_outliers = series.copy()

        rolling_median = series.rolling(
            window=window, center=True, min_periods=1
        ).median()
        rolling_std = series.rolling(window=window, center=True, min_periods=1).std()
        upper_bound = rolling_median + (sigma * rolling_std)
        lower_bound = rolling_median - (sigma * rolling_std)
        rolling_outliers = (series < lower_bound) | (series > upper_bound)

        print(f"  - Found {rolling_outliers.sum()} outliers.")
        series_no_outliers[rolling_outliers] = np.nan
        return series_no_outliers

    def load_and_prepare_data(self, csv_path: str) -> pd.Series:
        """Loads, preprocesses, and interpolates tidal level data.

        Assumes the CSV has "timestamp" and "tidal_level" columns.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            A preprocessed time series.

        Raises:
            FileNotFoundError: If the CSV file is not found.
        """
        try:
            df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)
            series = df["tidal_level"].copy()
            series.name = "hourly_avg_tidal_level"
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        print(f"Original data points: {len(series)}")
        print(f"Initial missing values: {series.isnull().sum()}")

        series_no_outliers = self._handle_outliers(series)
        print(
            "Missing values after outlier removal: "
            f"{series_no_outliers.isnull().sum()}"
        )

        series_filled = series_no_outliers.interpolate(method="linear", limit=24 * 3)
        if series_filled.isnull().sum() > 0:
            for i in range(1, 8):
                series_filled.fillna(series_filled.shift(24 * 7 * i), inplace=True)
                if series_filled.isnull().sum() == 0:
                    break
        series_filled.fillna(method="bfill", inplace=True)
        series_filled.fillna(method="ffill", inplace=True)

        self.original_data = series_filled
        print(f"Preprocessed data points: {len(self.original_data)}")
        return self.original_data

    def _decompose_with_stl(
        self,
        series: pd.Series,
        period: int = 365
    ) -> seasonal.DecomposeResult:
        """Performs time series decomposition using STL.

        Args:
            series: The time series data (hourly average).
            period: The seasonal period in days.

        Returns:
            A DecomposeResult object from statsmodels.
        """
        print("Performing STL decomposition...")
        daily_series = series.resample("D").mean().interpolate()
        if len(daily_series) < period * 2:
            period = min(365, len(daily_series) // 2)
            print(f"  - Data is short, adjusting period to {period} days.")

        stl = seasonal.STL(daily_series, period=period, robust=True)
        return stl.fit()

    def model_yearly_seasonality(
        self,
        series: pd.Series,
        period: int = 365,
        model: str = "additive"
    ) -> pd.Series:
        """Models the yearly seasonality and removes it from the series.

        Args:
            series: The input time series.
            period: The seasonal period in days.
            model: The decomposition model type ('additive' or 'multiplicative').

        Returns:
            A time series with the yearly seasonal component removed.
        """
        print(f"Modeling yearly seasonality ({period}-day period, {model} model)...")
        yearly_decomposition = self._decompose_with_stl(series, period)
        self.full_decomposition_result = yearly_decomposition

        pattern = yearly_decomposition.seasonal.dropna()
        self.yearly_seasonal_pattern = (
            pattern.groupby(pattern.index.dayofyear).median().to_dict()
        )
        if 366 not in self.yearly_seasonal_pattern:
            self.yearly_seasonal_pattern[366] = self.yearly_seasonal_pattern.get(1, 0)

        if model == "additive":
            return series - series.index.map(
                lambda dt: self.yearly_seasonal_pattern.get(dt.dayofyear, 0)
            )
        return series / series.index.map(
            lambda dt: self.yearly_seasonal_pattern.get(dt.dayofyear, 1)
        )

    def model_trend(self, trend_component: pd.Series, degree: int = 2):
        """Models the trend using polynomial regression.

        Args:
            trend_component: The trend component from decomposition.
            degree: The degree of the polynomial.
        """
        print(f"Modeling trend with a {degree}-degree polynomial...")
        trend_data = trend_component.dropna()

        q1, q3 = trend_data.quantile(0.25), trend_data.quantile(0.75)
        iqr = q3 - q1
        mask = (trend_data >= q1 - 1.5 * iqr) & (trend_data <= q3 + 1.5 * iqr)
        trend_data_clean = trend_data[mask]

        x_numeric = trend_data_clean.index.astype(np.int64) // 10**9
        y_values = trend_data_clean.values

        coeffs = np.polyfit(x_numeric, y_values, deg=degree)
        self.trend_model = np.poly1d(coeffs)

    def model_daily_seasonality_and_noise(
        self,
        series: pd.Series,
        period: int = 24,
        model: str = "additive"
    ):
        """Models daily seasonality and the remaining noise.

        Args:
            series: Time series with yearly seasonality removed.
            period: The daily seasonal period in hours.
            model: The decomposition model type.
        """
        print(f"Modeling daily seasonality and noise ({period}-hour period)...")
        daily_decomposition = seasonal.seasonal_decompose(
            series, model=model, period=period
        )
        seasonal_component = daily_decomposition.seasonal.dropna()
        self.daily_seasonal_pattern = seasonal_component.iloc[:period].values
        self._model_noise_ar(daily_decomposition.resid)

    def _model_noise_ar(self, residual_component: pd.Series, max_lags: int = 10):
        """Models the noise using an Autoregressive (AR) model.

        Args:
            residual_component: The residual component from decomposition.
            max_lags: The maximum number of lags to test for the AR model.
        """
        print("Modeling noise with AR model...")
        noise_data = residual_component.dropna()

        best_aic, best_lag = np.inf, 0
        for lag in range(1, max_lags + 1):
            try:
                model = ar_model.AutoReg(noise_data, lags=lag).fit()
                if model.aic < best_aic:
                    best_aic, best_lag = model.aic, lag
            except Exception:
                continue

        if best_lag == 0:
            print("  - Could not find optimal AR lag. Using normal distribution.")
            self._model_noise_normal(residual_component)
            return

        print(f"  - Optimal AR lag: {best_lag} (AIC: {best_aic:.2f})")
        ar_model_fit = ar_model.AutoReg(noise_data, lags=best_lag).fit()
        self.noise_model = {
            "type": "ar",
            "params": ar_model_fit.params,
            "std_resid": np.std(ar_model_fit.resid.dropna()),
            "lags": best_lag,
        }

    def _model_noise_normal(self, residual_component: pd.Series):
        """Models the noise using a normal distribution as a fallback."""
        noise_data = residual_component.dropna()
        self.noise_model = {
            "type": "normal",
            "mean": noise_data.mean(),
            "std": noise_data.std(),
        }

    def generate_trend(self, datetime_range: pd.DatetimeIndex) -> np.ndarray:
        """Generates trend values for a given date range."""
        if self.trend_model is None:
            raise ValueError("Trend model is not trained.")
        x_numeric = datetime_range.astype(np.int64) // 10**9
        return self.trend_model(x_numeric)

    def generate_yearly_seasonality(
        self,
        datetime_range: pd.DatetimeIndex
    ) -> np.ndarray:
        """Generates yearly seasonality values for a given date range."""
        if self.yearly_seasonal_pattern is None:
            raise ValueError("Yearly seasonality model is not trained.")
        day_of_year = datetime_range.dayofyear
        return np.array(
            [self.yearly_seasonal_pattern.get(doy, 0) for doy in day_of_year]
        )

    def generate_daily_seasonality(
        self,
        datetime_range: pd.DatetimeIndex
    ) -> np.ndarray:
        """Generates daily seasonality values for a given date range."""
        if self.daily_seasonal_pattern is None:
            raise ValueError("Daily seasonality model is not trained.")
        period = len(self.daily_seasonal_pattern)
        indices = datetime_range.hour % period
        return self.daily_seasonal_pattern[indices]

    def generate_noise(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Generates noise using the trained AR or normal model."""
        if self.noise_model is None:
            raise ValueError("Noise model is not trained.")

        np.random.seed(seed)

        if self.noise_model["type"] == "ar":
            lags = self.noise_model["lags"]
            params = self.noise_model["params"]
            std_resid = self.noise_model["std_resid"]
            noise = np.zeros(n_samples + lags)
            for t in range(lags, n_samples + lags):
                ar_component = np.dot(params[1:], noise[t - lags : t][::-1])
                random_shock = np.random.normal(0, std_resid)
                noise[t] = params[0] + ar_component + random_shock
            return noise[lags:]
        else:
            return np.random.normal(
                self.noise_model["mean"],
                self.noise_model["std"],
                n_samples,
            )

    def simulate(
        self,
        start_date: str,
        end_date: str,
        freq: str = "H",
        seed: Optional[int] = None,
    ) -> pd.Series:
        """Generates synthetic tidal level data for a given period.

        Args:
            start_date: The start date for the simulation (e.g., "2023-01-01").
            end_date: The end date for the simulation (e.g., "2023-12-31").
            freq: The frequency of the generated data (default: "H" for hourly).
            seed: A random seed for reproducibility.

        Returns:
            A pandas Series containing the simulated tidal level data.
        """
        datetime_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        trend = self.generate_trend(datetime_range)
        yearly = self.generate_yearly_seasonality(datetime_range)
        daily = self.generate_daily_seasonality(datetime_range)
        noise = self.generate_noise(len(datetime_range), seed=seed)
        simulated_level = trend + yearly + daily + noise
        return pd.Series(simulated_level, index=datetime_range, name="simulated_tidal_level")

    def run_analysis(
        self,
        csv_path: str,
        trend_degree: int = 2,
        decomposition_model: str = "additive",
    ) -> "TidalLevelSimulator":
        """Executes the full analysis pipeline.

        Args:
            csv_path: Path to the historical data file.
            trend_degree: The degree for the polynomial trend model.
            decomposition_model: The decomposition model type.

        Returns:
            The fitted TidalLevelSimulator instance.
        """
        print("1. Loading and preparing data...")
        series = self.load_and_prepare_data(csv_path)

        print("\n2. Modeling yearly seasonality...")
        series_no_yearly = self.model_yearly_seasonality(
            series, period=365, model=decomposition_model
        )

        print("\n3. Modeling trend...")
        if self.full_decomposition_result:
            self.model_trend(self.full_decomposition_result.trend, degree=trend_degree)

        print("\n4. Modeling daily seasonality and noise...")
        self.model_daily_seasonality_and_noise(
            series_no_yearly, period=24, model=decomposition_model
        )

        print("\nAnalysis complete.")
        return self
