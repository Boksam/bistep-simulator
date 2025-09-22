# Water Temperature Simulation and Analysis Web Application

This project is a web application that analyzes historical tide observation data (water temperature) to create a time-series model. It allows users to simulate water temperature for a specified period and visualize the results.

## Tech Stack

- **Backend**: Python, FastAPI, Pandas, Statsmodels, Matplotlib
- **Frontend**: React, TypeScript, Vite, Recharts, Axios
- **Package Manager**: pnpm

## Project Structure

```
/
├── backend/            # FastAPI backend server
│   ├── app.py          # API endpoint definitions
│   ├── plotting.py     # Chart generation for visualization
│   ├── water_temperature.py # Water temperature simulation logic
│   └── data/
│       ├── raw/        # Raw data (Excel files)
│       └── processed/  # Preprocessed data (CSV)
├── frontend/           # React frontend application
├── scripts/            # Data preprocessing scripts
└── README.md
```

## Installation and Setup

### 1. Prerequisites

- **Python 3.8+** and **Node.js 18+** must be installed.
- This project uses **pnpm** for frontend package management.
  ```bash
  npm install -g pnpm
  ```

### 2. Backend Setup

From the project root directory, run the following command to install the required Python packages.

```bash
pip install -r requirements.txt
```

### 3. Frontend Setup

Navigate to the `frontend` directory and run the following command to install the required Node.js packages.

```bash
cd frontend
pnpm install
```

### 4. Data Preprocessing

Before running the simulation, you need to process the raw data into a format suitable for analysis. Run the following script **only once** from the project root directory.

```bash
python scripts/preprocess_tide_data.py
```

This script reads the Excel files from `backend/data/raw/tide/`, calculates the hourly average water temperature, and saves the result to `backend/data/processed/hourly_avg_water_temperature.csv`.

### 5. Running the Application

#### 1. Start the Backend Server

In the project root directory, run the following command:

```bash
uvicorn backend.app:app --reload
```

The server will run at `http://127.0.0.1:8000`.

#### 2. Start the Frontend Server

Open a separate terminal, navigate to the `frontend` directory, and run the following command:

```bash
cd frontend
pnpm run dev
```

The development server will run at `http://localhost:5173`.

### 6. Usage

Open your web browser and go to `http://localhost:5173`. Select the desired start and end dates, then click the "Run Simulation" button. You will see the predicted water temperature data for that period and a comparison chart with the original data.
