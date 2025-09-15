"""
이 스크립트는 지정된 디렉토리에서 조위 관측 자료(Excel 파일)를 읽어와
수온 데이터를 전처리하고, 지정된 시간 주기(일/분/초)로 평균을 계산하여
CSV 파일로 저장합니다.
"""

import glob
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(ROOT_DIR, "assets", "meis.go.kr")
OUTPUT_DIR = os.path.join(ROOT_DIR, "assets")

RESAMPLE_FREQ = "H"  # 'T', 'S', 'D', 'H'

# NOTE: 파일 패턴에 `tide_202*.xlsx`로 고정되어 있으니 주의하십숑
FILE_PATTERN = "tide_202*.xlsx"
DATETIME_COL = "관측일시"
TEMPERATURE_COL = "수온"

# RESAMPLE_FREQ에 따라 동적으로 파일 이름 생성
FREQ_MAP = {"D": "daily", "T": "minute", "S": "second", "H": "hourly"}
OUTPUT_FILENAME = (
    f"{FREQ_MAP.get(RESAMPLE_FREQ, 'resampled')}_avg_water_temperature.csv"
)
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)


def find_file_paths(directory: str, pattern: str) -> list[str]:
    """지정된 디렉토리에서 특정 패턴과 일치하는 파일 경로를 찾습니다.

    Args:
        directory (str): 파일을 검색할 디렉토리 경로입니다.
        pattern (str): 파일 이름과 일치시킬 glob 패턴입니다.

    Returns:
        list[str]: 찾은 파일 경로의 정렬된 리스트입니다.

    Raises:
        FileNotFoundError: 일치하는 파일이 하나도 없을 경우 발생합니다.
    """
    file_pattern = os.path.join(directory, pattern)
    file_paths = sorted(glob.glob(file_pattern))
    if not file_paths:
        raise FileNotFoundError(
            f"No Excel files found in '{directory}' with pattern '{pattern}'. "
            "Please check the path and file names."
        )
    print(f"Found {len(file_paths)} files to process.")
    return file_paths


def load_and_combine_data(file_paths: list[str]) -> pd.DataFrame:
    """여러 Excel 파일을 읽어 하나의 데이터프레임으로 결합합니다.

    Args:
        file_paths (list[str]): 읽어올 Excel 파일 경로의 리스트입니다.

    Returns:
        pd.DataFrame: '관측일시'와 '수온' 컬럼을 포함하는 결합된 데이터프레임입니다.
                      데이터를 불러오지 못한 경우 빈 데이터프레임을 반환할 수 있습니다.
    """
    all_dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            if DATETIME_COL in df.columns and TEMPERATURE_COL in df.columns:
                all_dfs.append(df[[DATETIME_COL, TEMPERATURE_COL]])
            else:
                print(
                    f"Warning: '{DATETIME_COL}' or '{TEMPERATURE_COL}' column not found "
                    f"in {os.path.basename(file_path)}. Skipping this file."
                )
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")

    if not all_dfs:
        print("No data could be loaded.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임을 전처리합니다. (타입 변환, 결측치 처리, 인덱싱)

    Args:
        df (pd.DataFrame): 전처리할 원본 데이터프레임입니다.

    Returns:
        pd.DataFrame: 전처리된 데이터프레임입니다. '관측일시'가 인덱스로 설정됩니다.
    """
    df[TEMPERATURE_COL] = pd.to_numeric(df[TEMPERATURE_COL], errors="coerce")
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df.dropna(subset=[DATETIME_COL, TEMPERATURE_COL], inplace=True)
    df.set_index(DATETIME_COL, inplace=True)
    df.sort_index(inplace=True)
    return df


def resample_and_average(df: pd.DataFrame, freq: str) -> pd.Series:
    """데이터프레임을 지정된 주기로 리샘플링하고 평균을 계산합니다.

    Args:
        df (pd.DataFrame): '수온' 데이터를 포함하고 '관측일시'가 인덱스로 설정된 데이터프레임.
        freq (str): 리샘플링 주기 ('D', 'T', 'S' 등).

    Returns:
        pd.Series: 리샘플링된 평균 수온 데이터입니다.
    """
    return df[TEMPERATURE_COL].resample(freq).mean()


def save_data_to_csv(data: pd.Series, output_path: str):
    """데이터를 CSV 파일로 저장합니다.

    Args:
        data (pd.Series): 저장할 데이터 (Pandas Series).
        output_path (str): 저장할 CSV 파일의 전체 경로.
    """
    df_to_save = data.reset_index()
    df_to_save.columns = ["timestamp", "temperature"]
    df_to_save.loc[df_to_save["temperature"] == 0.0, "temperature"] = np.nan
    df_to_save.to_csv(output_path, header=True, index=False)
    print(f"\nSuccessfully processed and saved the data to:\n{output_path}")


def main():
    """메인 실행 함수. 데이터 처리 파이프라인을 총괄합니다."""
    try:
        file_paths = find_file_paths(DATA_DIR, FILE_PATTERN)
        combined_df = load_and_combine_data(file_paths)

        if combined_df.empty:
            print("Exiting because no data was loaded.")
            return

        processed_df = preprocess_data(combined_df)
        resampled_data = resample_and_average(processed_df, RESAMPLE_FREQ)
        save_data_to_csv(resampled_data, OUTPUT_CSV_PATH)

        print("\n--- Data Preview ---")
        print(resampled_data.head())
        print("...")
        print(resampled_data.tail())

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
