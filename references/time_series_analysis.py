import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

output_img_filename = "TimeSeriesAnalysis_Result.png"
output_excel_filename = "TimeSeriesAnalysis_Result.xlsx"

# 작업 디렉토리(cwd, pwd)를 소스코드가 저장된 경로로 변경
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 기본 한글 폰트
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# 1. 데이터 생성
np.random.seed(42)
date_rng = pd.date_range(start="2020-01-01", end="2022-12-31", freq="M")
trend = np.linspace(200, 700, len(date_rng))
seasonality = 90 * np.sin(2 * np.pi * date_rng.month / 12)
noise = np.random.normal(0, 40, len(date_rng))
ts_data = trend + seasonality + noise
ts = pd.Series(ts_data, index=date_rng)

# 2. 시계열 분해
result = seasonal_decompose(ts, model="additive", period=12)
trend_component = result.trend
seasonal_component = result.seasonal
residual_component = result.resid

# 3. Trend를 선형 회귀로 학습
notna_idx = trend_component.dropna().index
X_trend = np.arange(len(ts))[ts.index.isin(notna_idx)].reshape(-1, 1)
y_trend = trend_component.dropna().values
trend_lr = LinearRegression().fit(X_trend, y_trend)
trend_slope = trend_lr.coef_[0]
trend_intercept = trend_lr.intercept_
print(
    "Trend 선형회귀 결과: slope=%.4f, intercept=%.4f" % (trend_slope, trend_intercept)
)


# 4. Seasonality를 적합한 사인 함수로 학습
def seasonal_model(x, amp, phase, offset):
    return amp * np.sin(2 * np.pi * (x + phase) / 12) + offset


X_season = np.array([d.month for d in seasonal_component.index])
y_season = seasonal_component.values
popt, pcov = curve_fit(seasonal_model, X_season, y_season, p0=[90, 0, 0])
amp, phase, offset = popt
print(
    "Seasonality 사인 함수 파라미터: amplitude=%.4f, phase=%.4f, offset=%.4f"
    % (amp, phase, offset)
)

# 5. Noise를 정규분포로 모델링
noise_data = residual_component.dropna().values
noise_mean, noise_std = norm.fit(noise_data)
print("Noise 정규분포 파라미터: mean=%.4f, std_dev=%.4f" % (noise_mean, noise_std))

# 5.5. 결과를 엑셀 파일에 저장
with pd.ExcelWriter(output_excel_filename, engine="openpyxl") as writer:
    # Trend 선형회귀 결과
    trend_results = pd.DataFrame(
        {"파라미터": ["slope", "intercept"], "값": [trend_slope, trend_intercept]}
    )
    trend_results.to_excel(writer, sheet_name="Trend_선형회귀", index=False)

    # Seasonality 사인 함수 파라미터
    seasonal_results = pd.DataFrame(
        {"파라미터": ["amplitude", "phase", "offset"], "값": [amp, phase, offset]}
    )
    seasonal_results.to_excel(writer, sheet_name="Seasonality_사인함수", index=False)

    # Noise 정규분포 파라미터
    noise_results = pd.DataFrame(
        {"파라미터": ["mean", "std_dev"], "값": [noise_mean, noise_std]}
    )
    noise_results.to_excel(writer, sheet_name="Noise_정규분포", index=False)

print('결과가 "{}" 파일에 저장되었습니다.'.format(output_excel_filename))


# 6. 시각화 및 PNG 파일 저장
plt.figure(figsize=(12, 9))
plt.subplot(411)
plt.plot(ts, label="원본 시계열")
plt.legend(loc="upper left")
plt.subplot(412)
plt.plot(trend_component, label="추세 (Trend)", color="orange")
plt.legend(loc="upper left")
plt.subplot(413)
plt.plot(seasonal_component, label="계절성 (Seasonality)", color="green")
plt.legend(loc="upper left")
plt.subplot(414)
plt.plot(residual_component, label="노이즈 (Noise)", color="red")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(output_img_filename, dpi=300, bbox_inches="tight")
plt.close()  # 메모리 절약을 위해 figure 닫기
print('시각화 결과가 "{}" 파일로 저장되었습니다.'.format(output_img_filename))
