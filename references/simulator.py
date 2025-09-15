import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# 작업 디렉토리(cwd, pwd)를 소스코드가 저장된 경로로 변경
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 기본 한글 폰트
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


def load_analysis_results():
    """시계열 분석 결과를 엑셀 파일에서 로드"""
    excel_path = "../1. Time Series Analysis/TimeSeriesAnalysis_Result.xlsx"

    # Trend 선형회귀 결과 로드
    trend_df = pd.read_excel(excel_path, sheet_name="Trend_선형회귀")
    trend_slope = trend_df[trend_df["파라미터"] == "slope"]["값"].iloc[0]
    trend_intercept = trend_df[trend_df["파라미터"] == "intercept"]["값"].iloc[0]

    # Seasonality 사인 함수 파라미터 로드
    seasonal_df = pd.read_excel(excel_path, sheet_name="Seasonality_사인함수")
    amplitude = seasonal_df[seasonal_df["파라미터"] == "amplitude"]["값"].iloc[0]
    phase = seasonal_df[seasonal_df["파라미터"] == "phase"]["값"].iloc[0]
    offset = seasonal_df[seasonal_df["파라미터"] == "offset"]["값"].iloc[0]

    # Noise 정규분포 파라미터 로드
    noise_df = pd.read_excel(excel_path, sheet_name="Noise_정규분포")
    noise_mean = noise_df[noise_df["파라미터"] == "mean"]["값"].iloc[0]
    noise_std = noise_df[noise_df["파라미터"] == "std_dev"]["값"].iloc[0]

    return {
        "trend_slope": trend_slope,
        "trend_intercept": trend_intercept,
        "amplitude": amplitude,
        "phase": phase,
        "offset": offset,
        "noise_mean": noise_mean,
        "noise_std": noise_std,
    }


def generate_synthetic_data(params, start_date="2023-01-01", periods=60):
    """분석 결과를 활용하여 5년간의 합성 데이터 생성"""

    # 날짜 범위 생성 (월 단위)
    date_rng = pd.date_range(start=start_date, periods=periods, freq="M")

    # Trend 컴포넌트 생성 (선형 회귀 모델 사용)
    time_points = np.arange(len(date_rng))
    trend_component = params["trend_slope"] * time_points + params["trend_intercept"]

    # Seasonality 컴포넌트 생성 (사인 함수 모델 사용)
    def seasonal_model(x, amp, phase, offset):
        return amp * np.sin(2 * np.pi * (x + phase) / 12) + offset

    seasonal_component = seasonal_model(
        np.array([d.month for d in date_rng]),
        params["amplitude"],
        params["phase"],
        params["offset"],
    )

    # Noise 컴포넌트 생성 (정규분포 모델 사용)
    noise_component = np.random.normal(
        params["noise_mean"], params["noise_std"], len(date_rng)
    )

    # 전체 시계열 데이터 생성 (가법 모형)
    synthetic_data = trend_component + seasonal_component + noise_component

    # 결과를 DataFrame으로 정리
    result_df = pd.DataFrame(
        {
            "Date": date_rng,
            "Synthetic_Data": synthetic_data,
            "Trend": trend_component,
            "Seasonality": seasonal_component,
            "Noise": noise_component,
        }
    )

    return result_df


def visualize_synthetic_data(df, output_filename="Simulator_Result.png"):
    """생성된 합성 데이터 시각화 및 PNG 파일 저장"""

    plt.figure(figsize=(15, 10))

    # 1. 전체 합성 데이터
    plt.subplot(411)
    plt.plot(
        df["Date"],
        df["Synthetic_Data"],
        label="합성 시계열 데이터",
        color="blue",
        linewidth=2,
    )
    plt.title("5년간 생성된 합성 시계열 데이터", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    # 2. Trend 컴포넌트
    plt.subplot(412)
    plt.plot(df["Date"], df["Trend"], label="추세 (Trend)", color="orange", linewidth=2)
    plt.title("추세 컴포넌트", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    # 3. Seasonality 컴포넌트
    plt.subplot(413)
    plt.plot(
        df["Date"],
        df["Seasonality"],
        label="계절성 (Seasonality)",
        color="green",
        linewidth=2,
    )
    plt.title("계절성 컴포넌트", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    # 4. Noise 컴포넌트
    plt.subplot(414)
    plt.plot(df["Date"], df["Noise"], label="노이즈 (Noise)", color="red", linewidth=2)
    plt.title("노이즈 컴포넌트", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f'시각화 결과가 "{output_filename}" 파일로 저장되었습니다.')


def save_synthetic_data_to_excel(df, output_filename="Simulator_Result.xlsx"):
    """생성된 합성 데이터를 엑셀 파일로 저장"""

    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        # 전체 데이터 저장
        df.to_excel(writer, sheet_name="합성_시계열_데이터", index=False)

        # 통계 요약 정보 저장
        summary_stats = pd.DataFrame(
            {
                "통계량": ["평균", "표준편차", "최솟값", "최댓값", "중앙값"],
                "합성_데이터": [
                    df["Synthetic_Data"].mean(),
                    df["Synthetic_Data"].std(),
                    df["Synthetic_Data"].min(),
                    df["Synthetic_Data"].max(),
                    df["Synthetic_Data"].median(),
                ],
            }
        )
        summary_stats.to_excel(writer, sheet_name="통계_요약", index=False)

    print(f'합성 데이터가 "{output_filename}" 파일로 저장되었습니다.')


def main():
    """메인 실행 함수"""
    print("=== 시계열 분석 결과 기반 시뮬레이터 ===")

    try:
        # 1. 시계열 분석 결과 로드
        print("1. 시계열 분석 결과 로드 중...")
        params = load_analysis_results()
        print(
            "   - Trend 파라미터:",
            f"slope={params['trend_slope']:.4f}, intercept={params['trend_intercept']:.4f}",
        )
        print(
            "   - Seasonality 파라미터:",
            f"amplitude={params['amplitude']:.4f}, phase={params['phase']:.4f}, offset={params['offset']:.4f}",
        )
        print(
            "   - Noise 파라미터:",
            f"mean={params['noise_mean']:.4f}, std={params['noise_std']:.4f}",
        )

        # 2. 5년간의 합성 데이터 생성
        print("\n2. 5년간의 합성 데이터 생성 중...")
        synthetic_df = generate_synthetic_data(
            params, start_date="2023-01-01", periods=60
        )
        print(f"   - {len(synthetic_df)}개월의 데이터 생성 완료")
        print(
            f"   - 데이터 범위: {synthetic_df['Date'].min().strftime('%Y-%m')} ~ {synthetic_df['Date'].max().strftime('%Y-%m')}"
        )

        # 3. 시각화 및 PNG 파일 저장
        print("\n3. 시각화 및 PNG 파일 저장 중...")
        visualize_synthetic_data(synthetic_df)

        # 4. 엑셀 파일 저장
        print("\n4. 엑셀 파일 저장 중...")
        save_synthetic_data_to_excel(synthetic_df)

        print("\n=== 시뮬레이터 실행 완료 ===")
        print("생성된 파일:")
        print("- Simulator_Result.png (시각화 결과)")
        print("- Simulator_Result.xlsx (합성 데이터)")

    except FileNotFoundError:
        print("오류: 시계열 분석 결과 파일을 찾을 수 없습니다.")
        print(
            "먼저 '../1. Time Series Analysis/time_series_analysis.py'를 실행하여 분석 결과를 생성해주세요."
        )
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
