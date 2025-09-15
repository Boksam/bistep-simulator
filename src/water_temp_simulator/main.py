import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa import ar_model, seasonal


class WaterTemperatureSimulator:
    """과거 수온 데이터를 기반으로 시계열을 분해하고, 이를 바탕으로 미래의 수온 데이터를 시뮬레이션하는 클래스.

    추세(Trend), 연간 계절성(Yearly Seasonality), 일일 계절성(Daily Seasonality), 노이즈(Noise) 등의 구성요소를 분리 및 모델링합니다.
    현실적인 변동성을 위해 노이즈(잔차)를 자기회귀(AR) 모델로 모델링합니다.
    """

    def __init__(self):
        """시뮬레이터 초기화"""
        self.trend_model: Optional[np.poly1d] = None
        self.yearly_seasonal_pattern: Optional[Dict[int, float]] = None
        self.daily_seasonal_pattern: Optional[np.ndarray] = None
        self.noise_model: Optional[Dict] = None
        self.original_data: Optional[pd.Series] = None
        self.decomposed_df: Optional[pd.DataFrame] = None
        self.full_decomposition_result: Optional[seasonal.DecomposeResult] = None

    def handle_outliers(
        self,
        series: pd.Series,
        window: int = 24 * 7,
        sigma: float = 2.5,
        absolute_threshold: float = 30.0,
    ) -> pd.Series:
        """이상치를 탐지하고 처리합니다.

        다음과 같은 방법으로 이상치를 탐지합니다:
            1. 절대 임계값 초과: 수온이 절대 임계값(기본값: 30)을 초과하는 경우
            2. 롤링 윈도우 기반 탐지: 지정된 윈도우 내에서 중앙값과 표준편차를 계산하여, sigma 표준편차를 벗어나는 값을 이상치로 간주

        Args:
            series: 이상치를 탐지할 시계열 데이터
            window: 롤링 윈도우 크기 (기본값: 1주일)
            sigma: 표준편차 기준 (기본값: 2.5)
            absolute_threshold: 절대 임계값 (기본값: 30.0도)

        Returns:
            이상치가 NaN으로 처리된 시계열 데이터
        """
        print(f"이상치 탐지 시작...")

        series_no_outliers = series.copy()

        # 절대 임계값을 초과하는 값은 이상치로 간주
        absolute_outliers = series > absolute_threshold

        # 롤링 윈도우 기반 이상치 탐지
        # 윈도우 내의 중앙값을 계산하고, 이를 기준으로 sigma 표준편차를 벗어나는 값을 이상치로 간주
        rolling_median = series.rolling(
            window=window, center=True, min_periods=1
        ).median()
        rolling_std = series.rolling(window=window, center=True, min_periods=1).std()

        upper_bound = rolling_median + (sigma * rolling_std)
        lower_bound = rolling_median - (sigma * rolling_std)

        rolling_outliers = (series < lower_bound) | (series > upper_bound)

        # 모든 방법을 종합한 최종 이상치
        total_outliers = absolute_outliers | rolling_outliers

        print(
            f"절대 임계값({absolute_threshold}°C) 초과 이상치: {absolute_outliers.sum()}개"
        )
        print(f"롤링 윈도우 기반 이상치: {rolling_outliers.sum()}개")
        print(f"최종 탐지된 이상치 개수: {total_outliers.sum()}개")

        series_no_outliers[total_outliers] = np.nan

        return series_no_outliers

    def load_and_prepare_data(self, csv_path: str) -> pd.Series:
        """데이터를 로드하고 이상치 처리 후, 시계열 보간법으로 결측치를 처리합니다.

        시계열 데이터는 "timestamp", "temperature" 컬럼을 가지고 있다고 가정합니다.
        `handle_outliers` 메서드를 사용하여 이상치를 처리합니다.
        그 후 아래의 방법으로 결측치를 처리합니다:
            1. 선형 보간법 (limit=72, 즉 최대 3일치 결측치까지 보간)
            2. 긴 결측 구간은 같은 시간대의 과거 데이터로 채우기 (최대 1주일 전까지)
            3. 그래도 남은 결측치는 앞뒤로 채우기 (bfill, ffill)

        Args:
            csv_path: CSV 파일 경로

        Returns:
            전처리된 시계열 데이터

        Raises:
            FileNotFoundError: 지정된 경로에 CSV 파일이 없는 경우
        """
        try:
            df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)
            series = df["temperature"].copy()
            series.name = "시간별 평균 수온"
        except FileNotFoundError:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {csv_path}")

        print(f"원본 데이터 개수: {len(series)}")
        print(f"데이터 기간: {series.index.min()} ~ {series.index.max()}")
        print(f"초기 결측값 개수: {series.isnull().sum()}")
        print(f"데이터 통계: 평균={series.mean():.2f}°C, 표준편차={series.std():.2f}°C")
        print(f"최소값={series.min():.2f}°C, 최대값={series.max():.2f}°C")

        series_no_outliers = self.handle_outliers(
            series,
            window=24 * 7,  # 1주일
            sigma=2.5,
            absolute_threshold=30.0,  # 30도 이상은 이상치로 간주1
        )
        print(f"이상치 처리 후 결측값 개수: {series_no_outliers.isnull().sum()}")

        # 1. 선형 보간 (최대 3일치 결측치까지 보간)
        series_filled = series_no_outliers.interpolate(
            method="linear",
            limit=24 * 3,
        )

        # 2. 긴 결측 구간은 같은 시간대의 과거 데이터로 채우기 (최대 1주일 전까지)
        if series_filled.isnull().sum() > 0:
            for i in range(1, 8):
                shift_hours = 24 * 7 * i
                series_filled = series_filled.fillna(series_filled.shift(shift_hours))
                if series_filled.isnull().sum() == 0:
                    break

            # 그래도 남은 결측치는 앞뒤로 채우기
            series_filled = series_filled.fillna(method="bfill").fillna(method="ffill")

        self.original_data = series_filled
        print(f"전처리 후 데이터 개수: {len(self.original_data)}")
        print(
            f"전처리 후 통계: 평균={self.original_data.mean():.2f}°C, 표준편차={self.original_data.std():.2f}°C"
        )

        return self.original_data

    def decompose_with_stl(
        self,
        series: pd.Series,
        period: int = 365,
    ) -> seasonal.DecomposeResult:
        """시계열 분해를 수행합니다.

        장기적인 패턴을 더 잘 포착하기 위해 Hour -> Daily 리샘플링 후 분해를 수행합니다.
        1년 주기의 계절성을 포착하기 위해 최소 2년치의 데이터가 필요합니다. 만약 데이터가 부족한 경우,
        주기를 데이터 길이에 맞게 조정합니다. (하지만 정확도가 떨어질 수 있습니다.)

        시계열 분해에는 Additive Model을 사용합니다. y(t) = Trend + Seasonality + Residual

        Args:
            series: 시계열 데이터 (시간별 평균 수온)
            period: 계절성 주기 (기본값: 365일)

        Returns:
            분해 결과 (statsmodels의 DecomposeResult 객체)
        """
        print("\n시계열 분해 수행 중...")

        # 일일 평균으로 리샘플링
        daily_series = series.resample("D").mean().interpolate()

        # 최소 2년치 데이터가 있어야 365일 주기 분석 가능
        if len(daily_series) < period * 2:
            period = min(365, len(daily_series) // 2)
            print(f"데이터 길이 제한으로 주기를 {period}일로 조정")

        # 분해 수행 (multiplicative 모델도 시도해볼 수 있음)
        decomposition = seasonal.seasonal_decompose(
            daily_series,
            model="additive",
            period=period,
            extrapolate_trend=0,
        )

        return decomposition

    def model_yearly_seasonality(
        self,
        series: pd.Series,
        period: int = 365,
        model: str = "additive",
    ) -> pd.Series:
        """연간 계절성을 모델링하고 원본 데이터에서 분리합니다.

        장기적인 패턴을 더 잘 포착하기 위해 Hour -> Daily 리샘플링 후 분해를 수행합니다.
        1년 주기의 계절성을 포착하기 위해 최소 2년치의 데이터가 필요합니다. 만약 데이터가 부족한 경우,
        주기를 데이터 길이에 맞게 조정합니다. (하지만 정확도가 떨어질 수 있습니다.)

        덧셈 모델(additive) 인 경우, y(t) = Trend + Seasonality + Residual 형태로 분해하고,
        곱셈 모델(multiplicative) 인 경우, y(t) = Trend * Seasonality * Residual 형태로 분해합니다.

        Args:
            series: 시계열 데이터 (시간별 평균 수온)
            period: 계절성 주기 (기본값: 365일)
            model: 분해 모델 유형 ("additive" 또는 "multiplicative", 기본값: "additive")

        Returns:
            연간 계절성이 제거된 시계열 데이터
        """
        print(f"연간 계절성 모델링 ({period}일 주기, {model} 모델)...")

        # 일일 평균 데이터로 변환
        daily_series = series.resample("D").mean().interpolate()

        if len(daily_series) < period * 2:
            period = len(daily_series) // 2
            print(f"데이터 길이가 짧아 연간 주기를 {period}일로 조정합니다.")

        # STL 분해를 사용하여 연간 계절성 추출
        yearly_decomposition = self.decompose_with_stl(series, period)
        self.full_decomposition_result = yearly_decomposition

        # 계절성 패턴 저장 (이상치 영향 최소화)
        pattern = yearly_decomposition.seasonal.dropna()

        # 각 날짜(1~365)에 대한 중앙값 계산
        self.yearly_seasonal_pattern = (
            pattern.groupby(pattern.index.dayofyear).median().to_dict()
        )

        # 윤년 처리
        if (
            366 not in self.yearly_seasonal_pattern
            and 1 in self.yearly_seasonal_pattern
        ):
            self.yearly_seasonal_pattern[366] = self.yearly_seasonal_pattern[1]

        print("연간 계절성 모델링 완료.")

        # 원본 시계열에서 연간 계절성 제거
        if model == "additive":
            # 덧셈 모델인 경우, 계절성 패턴을 빼줍니다.
            return series - series.index.map(
                lambda dt: self.yearly_seasonal_pattern.get(dt.dayofyear, 0)
            )
        else:
            # 곱셈 모델인 경우, 계절성 패턴으로 나눠줍니다.
            return series / series.index.map(
                lambda dt: self.yearly_seasonal_pattern.get(dt.dayofyear, 1)
            )

    def model_trend(
        self,
        trend_component: pd.Series,
        degree: int = 2,
    ) -> np.poly1d:
        """다항 회귀를 사용하여 추세를 모델링합니다.

        IQR 방법으로 이상치를 제거하고, 2차 다항식을 사용하여 추세(Trend)를 다항 회귀로 모델링합니다.

        Args:
            trend_component: 추세 성분 (시계열 분해 결과의 trend)
            degree: 다항식 차수 (기본값: 2차)

        Returns:
            추세 모델 (numpy의 poly1d 객체)
        """
        print(f"추세 모델링 ({degree}차 다항식)...")
        trend_data = trend_component.dropna()

        # 혹시 남아있을 이상치를 제거하기 위해 IQR 방법을 적용합니다.
        Q1 = trend_data.quantile(0.25)
        Q3 = trend_data.quantile(0.75)
        IQR = Q3 - Q1
        mask = (trend_data >= Q1 - 1.5 * IQR) & (trend_data <= Q3 + 1.5 * IQR)
        trend_data_clean = trend_data[mask]

        # 여름에 덥고 겨울에 추운 패턴을 잘 포착하기 위해 2차원 다항식 사용
        # TODO: 3차 이상도 시도해보고 overfitting 여부 확인
        x_numeric = trend_data_clean.index.astype(np.int64) // 10**9
        y_values = trend_data_clean.values

        coeffs = np.polyfit(x_numeric, y_values, deg=degree)
        self.trend_model = np.poly1d(coeffs)

        y_pred = self.trend_model(x_numeric)
        r2 = 1 - np.sum((y_values - y_pred) ** 2) / np.sum(
            (y_values - np.mean(y_values)) ** 2
        )
        print(f"추세 모델 R² 점수: {r2:.4f}")
        print(f"추세 범위: {y_values.min():.2f}°C ~ {y_values.max():.2f}°C")

        return self.trend_model

    def model_daily_seasonality_and_noise(
        self,
        series: pd.Series,
        period: int = 24,
        model: str = "additive",
    ) -> None:
        """일일 계절성과 노이즈(잔차)를 모델링합니다.

        24시간 주기로 반복되는 일일 계절성을 추출하고, 잔차를 자기회귀(AR) 모델로 모델링합니다.

        Args:
            series: 연간 계절성이 제거된 시계열 데이터
            period: 일일 계절성 주기 (기본값: 24시간)
            model: 분해 모델 유형 ("additive" 또는 "multiplicative", 기본값: "additive")
        """
        print(f"일일 계절성 및 노이즈 모델링 ({period}시간 주기, {model} 모델)...")
        daily_decomposition = seasonal.seasonal_decompose(
            series, model=model, period=period
        )

        seasonal_component = daily_decomposition.seasonal.dropna()
        self.daily_seasonal_pattern = seasonal_component.iloc[:period].values

        self.model_noise_ar(daily_decomposition.resid)
        print("일일 계절성 및 노이즈 모델링 완료.")

    def model_noise_ar(
        self,
        residual_component: pd.Series,
        max_lags: int = 10,
    ) -> Dict:
        """잔차(노이즈)를 자기회귀(AR) 모델로 모델링하여 현실성을 높입니다.

        lags를 1부터 max_lags까지 변화시키며 AIC 값을 계산하여 최적의 lag를 선택합니다.
        최적의 lag를 찾지 못한 경우, 정규분포를 이용한 기본 노이즈 모델링을 수행합니다.

        Args:
            residual_component: 잔차 성분 (시계열 분해 결과의 resid)
            max_lags: 최대 lag 수 (기본값: 10)

        Returns:
            노이즈 모델 정보 (딕셔너리)
        """
        print("현실적인 노이즈 모델링 (AR 모델)...")
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
            print("최적 AR lag를 찾지 못했습니다. 기본 정규분포 모델을 사용합니다.")
            return self.model_noise_normal(residual_component)

        print(f"최적 AR lag: {best_lag} (AIC: {best_aic:.2f})")
        ar_model_fit = ar_model.AutoReg(noise_data, lags=best_lag).fit()

        self.noise_model = {
            "type": "ar",
            "params": ar_model_fit.params,
            "std_resid": np.std(ar_model_fit.resid.dropna()),
            "lags": best_lag,
        }
        return self.noise_model

    def model_noise_normal(
        self,
        residual_component: pd.Series,
    ) -> Dict:
        """정규 분포를 이용한 기본 노이즈 모델링을 수행합니다."""
        noise_data = residual_component.dropna()
        self.noise_model = {
            "type": "normal",
            "mean": noise_data.mean(),
            "std": noise_data.std(),
        }
        print(
            f"잔차 통계 (정규분포): 평균={self.noise_model['mean']:.4f}, 표준편차={self.noise_model['std']:.4f}"
        )
        return self.noise_model

    def generate_trend(
        self,
        datetime_range: pd.DatetimeIndex,
    ) -> np.ndarray:
        """주어진 날짜 범위에 대한 추세 값을 생성합니다."""
        if self.trend_model is None:
            raise ValueError("추세 모델이 훈련되지 않았습니다.")
        x_numeric = datetime_range.astype(np.int64) // 10**9
        return self.trend_model(x_numeric)

    def generate_yearly_seasonality(
        self,
        datetime_range: pd.DatetimeIndex,
    ) -> np.ndarray:
        """연간 계절성 값을 생성합니다.

        Args:
            datetime_range: 날짜 범위 (DatetimeIndex)

        Returns:
            연간 계절성 값 (numpy 배열)
        """
        if self.yearly_seasonal_pattern is None:
            raise ValueError("연간 계절성 모델이 훈련되지 않았습니다.")
        day_of_year = datetime_range.dayofyear
        return np.array(
            [self.yearly_seasonal_pattern.get(doy, 0) for doy in day_of_year]
        )

    def generate_daily_seasonality(
        self,
        datetime_range: pd.DatetimeIndex,
    ) -> np.ndarray:
        """일일 계절성 값을 생성합니다.

        Args:
            datetime_range: 날짜 범위 (DatetimeIndex)

        Returns:
            일일 계절성 값 (numpy 배열)
        """
        if self.daily_seasonal_pattern is None:
            raise ValueError("일일 계절성 모델이 훈련되지 않았습니다.")
        period = len(self.daily_seasonal_pattern)
        indices = datetime_range.hour % period
        return self.daily_seasonal_pattern[indices]

    def generate_noise(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """학습된 AR 모델 또는 정규분포를 이용해 노이즈를 생성합니다.

        Args:
            n_samples: 생성할 노이즈 샘플 수
            seed: 랜덤 시드 (재현성을 위해 설정 가능, 기본값: None)

        Returns:
            생성된 노이즈 값 (numpy 배열)
        """
        if self.noise_model is None:
            raise ValueError("노이즈 모델이 훈련되지 않았습니다.")

        if seed is not None:
            np.random.seed(seed)

        if self.noise_model["type"] == "ar":
            # AR 모델을 사용하여 노이즈 생성
            lags, params, std_resid = (
                self.noise_model["lags"],
                self.noise_model["params"],
                self.noise_model["std_resid"],
            )
            noise = np.zeros(n_samples + lags)
            for t in range(lags, n_samples + lags):
                ar_component = np.dot(params[1:], noise[t - lags : t][::-1])
                random_shock = np.random.normal(0, std_resid)
                noise[t] = params[0] + ar_component + random_shock
            return noise[lags:]
        else:
            # 정규분포를 사용하여 노이즈 생성
            return np.random.normal(
                self.noise_model["mean"], self.noise_model["std"], n_samples
            )

    def simulate_data(
        self,
        start_date: str,
        end_date: str,
        freq: str = "H",
        seed: Optional[int] = None,
    ) -> pd.Series:
        """분해된 구성요소를 다시 조합하여 특정 기간의 데이터를 시뮬레이션합니다.

        Args:
            start_date: 시뮬레이션 시작 날짜 (예: "2023-01-01")
            end_date: 시뮬레이션 종료 날짜 (예: "2023-12-31")
            freq: 시뮬레이션 주기 (기본값: "H" - 시간별)
            seed: 랜덤 시드 (재현성을 위해 설정 가능, 기본값: None)

        Returns:
            시뮬레이션된 수온 데이터 (Pandas Series)
        """
        datetime_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        trend = self.generate_trend(datetime_range)
        yearly = self.generate_yearly_seasonality(datetime_range)
        daily = self.generate_daily_seasonality(datetime_range)
        noise = self.generate_noise(len(datetime_range), seed=seed)
        simulated_temp = trend + yearly + daily + noise
        return pd.Series(simulated_temp, index=datetime_range, name="시뮬레이션_수온")

    def analyze(
        self,
        csv_path: str,
        trend_degree: int = 2,
        decomposition_model: str = "additive",
    ) -> "WaterTemperatureSimulator":
        """데이터 로드부터 각 구성요소 모델링까지 전체 분석 파이프라인을 실행합니다.

        Args:
            csv_path: CSV 파일 경로
            trend_degree: 추세 모델링에 사용할 다항식 차수 (기본값: 2차)
            decomposition_model: 시계열 분해 모델 유형 ("additive" 또는 "multiplicative", 기본값: "additive")

        Returns:
            self: 분석이 완료된 WaterTemperatureSimulator 객체
        """
        print("\n1. 데이터 로드 및 전처리...")
        series = self.load_and_prepare_data(csv_path)

        print("\n2. 연간 계절성 분리...")
        series_no_yearly = self.model_yearly_seasonality(
            series, period=365, model=decomposition_model
        )

        print("\n3. 추세 모델링...")
        if self.full_decomposition_result:
            self.model_trend(self.full_decomposition_result.trend, degree=trend_degree)

        print("\n4. 일일 계절성 및 노이즈 모델링...")
        self.model_daily_seasonality_and_noise(
            series_no_yearly, period=24, model=decomposition_model
        )

        return self


def plot_decomposition_results(
    decomposition_result: seasonal.DecomposeResult, output_dir: str
):
    """statsmodels의 seasonal_decompose 결과를 시각화하고 파일로 저장합니다."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle("시계열 분해 결과 (일일 평균 데이터 기준, 이상치 제거)", fontsize=18)

    components_map = {
        "Observed": "observed",
        "Trend": "trend",
        "Seasonal": "seasonal",
        "Residual": "resid",
    }

    colors = ["cornflowerblue", "tomato", "forestgreen", "gray"]

    for i, ((name, attr_name), color) in enumerate(zip(components_map.items(), colors)):
        component_data = getattr(decomposition_result, attr_name)

        plot_kwargs = {"color": color, "legend": False}
        if name == "Residual":
            plot_kwargs.update({"linestyle": "None", "marker": "."})

        component_data.plot(ax=axes[i], **plot_kwargs)
        axes[i].set_ylabel(name)
        axes[i].set_title(f"{name} Component", fontsize=12)
        axes[i].grid(True, linestyle="--", alpha=0.6)

        # 각 컴포넌트의 통계 정보 추가
        if not component_data.isnull().all():
            stats_text = (
                f"Mean: {component_data.mean():.2f}, Std: {component_data.std():.2f}"
            )
            axes[i].text(
                0.02,
                0.95,
                stats_text,
                transform=axes[i].transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    axes[-1].set_xlabel("Date")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plot_path = os.path.join(output_dir, "time_series_decomposition_improved.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n시계열 분해 그래프 저장 완료: {plot_path}")
    plt.show()


def plot_comparison(
    original_series: pd.Series, simulated_series: pd.Series, output_dir: str
):
    """원본과 시뮬레이션 데이터를 비교하는 개선된 시각화"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # 1. 전체 시계열 비교
    axes[0].plot(
        original_series.index,
        original_series.values,
        label="Original (Outliers Removed)",
        color="cornflowerblue",
        alpha=0.7,
    )
    axes[0].plot(
        simulated_series.index,
        simulated_series.values,
        label="Simulated",
        color="tomato",
        alpha=0.7,
    )
    axes[0].set_title("전체 시계열 비교", fontsize=14)
    axes[0].set_ylabel("Water Temperature (°C)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 월별 평균 비교
    original_monthly = original_series.resample("M").mean()
    simulated_monthly = simulated_series.resample("M").mean()

    axes[1].plot(
        original_monthly.index,
        original_monthly.values,
        marker="o",
        label="Original Monthly Avg",
        color="cornflowerblue",
    )
    axes[1].plot(
        simulated_monthly.index,
        simulated_monthly.values,
        marker="s",
        label="Simulated Monthly Avg",
        color="tomato",
    )
    axes[1].set_title("월별 평균 수온 비교", fontsize=14)
    axes[1].set_ylabel("Water Temperature (°C)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 분포 비교 (히스토그램)
    axes[2].hist(
        original_series.values,
        bins=50,
        alpha=0.5,
        label="Original",
        color="cornflowerblue",
        density=True,
    )
    axes[2].hist(
        simulated_series.values,
        bins=50,
        alpha=0.5,
        label="Simulated",
        color="tomato",
        density=True,
    )
    axes[2].set_title("수온 분포 비교", fontsize=14)
    axes[2].set_xlabel("Water Temperature (°C)")
    axes[2].set_ylabel("Density")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "comparison_analysis.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n비교 분석 그래프 저장 완료: {plot_path}")
    plt.show()


def main():
    """메인 실행 함수"""

    csv_path = "./hourly_avg_water_temperature.csv"
    output_dir = "./figures"
    os.makedirs(output_dir, exist_ok=True)

    # --- 실행 ---
    try:
        # 1. 분석 실행
        simulator = WaterTemperatureSimulator()
        simulator.analyze(csv_path, trend_degree=2, decomposition_model="additive")

        # 2. 시계열 분해 결과 시각화
        if simulator.full_decomposition_result:
            plot_decomposition_results(simulator.full_decomposition_result, output_dir)

        # 3. 데이터 시뮬레이션
        # 원본 데이터와 동일한 기간으로 시뮬레이션
        if simulator.original_data is not None:
            start_date = simulator.original_data.index.min().strftime("%Y-%m-%d")
            end_date = simulator.original_data.index.max().strftime("%Y-%m-%d")

            simulated_data = simulator.simulate_data(start_date, end_date, seed=42)

            print("시뮬레이션 데이터 생성 완료!")
            print(
                f"시뮬레이션 통계: 평균={simulated_data.mean():.2f}°C, 표준편차={simulated_data.std():.2f}°C"
            )
            print(
                f"원본 통계: 평균={simulator.original_data.mean():.2f}°C, 표준편차={simulator.original_data.std():.2f}°C"
            )

            excel_file_path = os.path.join(
                output_dir, "simulated_water_temperature.xlsx"
            )
            simulated_data.to_excel(excel_file_path)
            print(f"\n시뮬레이션 데이터를 Excel 파일로 저장했습니다: {excel_file_path}")

            # 4. 개선된 비교 시각화
            plot_comparison(simulator.original_data, simulated_data, output_dir)

    except FileNotFoundError as e:
        print("데이터 파일을 찾을 수 없습니다: %s", e)
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
