# Bistep 수온 시뮬레이션 프로젝트

## 1. 개요

이 프로젝트는 해양수산부 국립해양조사원(meis.go.kr)에서 제공하는 조위 데이터를 기반으로 수온을 시뮬레이션하고 분석합니다. 시계열 분석 기법을 활용하여 수온 변화를 예측하고 시각화 자료를 생성하는 것을 목표로 합니다.

## 2. 디렉토리 구조

```
bistep/
├── assets/meis.go.kr/
│   └── tide_*.xlsx            # 국립해양조사원 원본 조위 데이터
├── src/
│   ├── generate_csv.py         # 원본 데이터를 병합하고 전처리하여 CSV 파일 생성
│   ├── requirements.txt        # 프로젝트 의존성 목록
│   └── water_temp_simulator/
│       ├── main.py             # 시계열 분석 및 수온 시뮬레이션 메인 스크립트
│       ├── hourly_avg_water_temperature.csv # 전처리된 시간별 평균 수온 데이터
│       └── figures/            # 시뮬레이션 결과 그래프 및 데이터
├── references/                 # 개발 참고용 스크립트
└── README.md                   # 프로젝트 설명 파일
```

- **`assets/meis.go.kr/`**: 국립해양조사원에서 다운로드한 원본 조위 데이터 (.xlsx) 파일들이 위치합니다.
- **`src/`**: 소스 코드 디렉토리입니다.
    - `generate_csv.py`: `assets` 폴더의 엑셀 파일들을 병합하고 전처리하여 `hourly_avg_water_temperature.csv` 파일을 생성합니다.
    - `water_temp_simulator/main.py`: 전처리된 데이터를 바탕으로 시계열 분석 및 수온 시뮬레이션을 수행하는 메인 스크립트입니다.
    - `water_temp_simulator/figures/`: 시뮬레이션 결과 및 분석 과정에서 생성된 그래프 이미지와 결과 파일이 저장됩니다.
- **`references/`**: 프로젝트 개발에 참고한 코드나 초기 버전의 스크립트가 포함되어 있습니다.

## 3. 설치 및 실행 방법

### 3.1. 가상 환경 및 의존성 설치

프로젝트 실행을 위해 먼저 가상 환경을 활성화하고 필요한 라이브러리를 설치합니다.

```bash
# 가상 환경 활성화 (macOS/Linux)
source venv/bin/activate

# 필요한 라이브러리 설치
pip install -r src/requirements.txt
```

### 3.2. 데이터 전처리

`assets` 폴더에 있는 원본 `.xlsx` 파일들을 단일 CSV 파일로 병합하고 전처리합니다.

```bash
python src/generate_csv.py
```

이 스크립트를 실행하면 `src/water_temp_simulator/` 디렉토리에 `hourly_avg_water_temperature.csv` 파일이 생성됩니다.

### 3.3. 시뮬레이션 실행

전처리된 데이터를 사용하여 수온 시뮬레이션을 실행합니다.

```bash
python src/water_temp_simulator/main.py
```

실행이 완료되면 `src/water_temp_simulator/figures/` 디렉토리에 분석 그래프(`comparison_analysis.png`, `time_series_decomposition_improved.png`)와 시뮬레이션 결과(`simulated_water_temperature.xlsx`)가 저장됩니다.

## 4. 기술 스택 및 의존성

- Python 3.13
- 주요 라이브러리:
    - `pandas`: 데이터 조작 및 분석
    - `openpyxl`: Excel 파일 읽기
    - `statsmodels`: 시계열 분석

전체 의존성 목록은 `src/requirements.txt` 파일을 참고하십시오.
