"""
FastAPI server for providing simulated sensor data.
"""

import base64
import io
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from plotting import plot_comparison, plot_decomposition
from pydantic import BaseModel
from salinity import SalinitySimulator
from water_temperature import WaterTemperatureSimulator

PROCESSED_WATER_TEMP_PATH = os.path.join(
    os.path.dirname(__file__), "data", "processed", "hourly_avg_water_temperature.csv"
)
PROCESSED_SALINITY_PATH = os.path.join(
    os.path.dirname(__file__), "data", "processed", "hourly_avg_water_salinity.csv"
)


# Create a FastAPI app instance
app = FastAPI(
    title="Sensor Simulation API",
    description="Provides simulated time-series data for various sensors.",
    version="1.0.0",
)

# --- CORS Middleware ---
# Allow requests from our frontend development server
origins = [
    "http://localhost:5173",  # Vite default port
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class SimulationRequest(BaseModel):
    """Request model for simulation endpoints."""

    start_date: str
    end_date: str


class PlotResponse(BaseModel):
    """Response model for plots."""

    decomposition: str
    comparison: str


class SimulationResponse(BaseModel):
    """Response model for simulation data and plots."""

    simulation_data: List[Dict]
    plots: PlotResponse


# --- Application State ---
# A dictionary to hold our trained simulator instances
simulators: Dict[str, WaterTemperatureSimulator | SalinitySimulator] = {}


# --- Helper Functions ---
def fig_to_base64(fig) -> str:
    """Converts a Matplotlib figure to a Base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# --- Events ---
@app.on_event("startup")
async def startup_event():
    """Event handler for server startup.

    Loads and trains simulation models to avoid re-training on every API call.
    """
    print("Server is starting up...")

    # Load and train water temperature simulator
    print("Loading and training water temperature simulator...")
    try:
        water_temp_simulator = WaterTemperatureSimulator()
        water_temp_simulator.run_analysis(PROCESSED_WATER_TEMP_PATH, trend_degree=2)
        simulators["water_temperature"] = water_temp_simulator
        print("Water temperature simulator loaded successfully.")
    except FileNotFoundError:
        print("ERROR: Could not load water temperature simulator.")
        print(f"Data file not found at {PROCESSED_WATER_TEMP_PATH}.")
    except Exception as e:
        print(
            f"An unexpected error occurred during water temperature simulator loading: {e}"
        )

    # Load and train salinity simulator
    print("\nLoading and training salinity simulator...")
    try:
        salinity_simulator = SalinitySimulator()
        salinity_simulator.run_analysis(PROCESSED_SALINITY_PATH, trend_degree=0)
        simulators["salinity"] = salinity_simulator
        print("Salinity simulator loaded successfully.")
    except FileNotFoundError:
        print("ERROR: Could not load salinity simulator.")
        print(f"Data file not found at {PROCESSED_SALINITY_PATH}.")
    except Exception as e:
        print(f"An unexpected error occurred during salinity simulator loading: {e}")


# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint providing basic information about the API."""
    return {
        "message": "Welcome to the Sensor Simulation API!",
        "available_simulators": list(simulators.keys()),
        "documentation": "/docs",
    }


@app.post("/simulations/{sensor_name}", response_model=SimulationResponse)
async def run_simulation(sensor_name: str, request: SimulationRequest):
    """Runs a new simulation and returns the data along with visualization plots."""
    if sensor_name not in simulators:
        raise HTTPException(
            status_code=404,
            detail=f"Simulator for '{sensor_name}' not found or failed to load.",
        )

    try:
        datetime.strptime(request.start_date, "%Y-%m-%d")
        datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Please use YYYY-MM-DD."
        )

    simulator = simulators[sensor_name]
    print(
        f"Generating simulation for '{sensor_name}' from {request.start_date} to {request.end_date}..."
    )

    # 1. Generate simulated data
    simulated_series = simulator.simulate(
        start_date=request.start_date, end_date=request.end_date
    )
    result_df = simulated_series.reset_index()
    result_df.columns = ["timestamp", "value"]
    json_data = result_df.to_dict(orient="records")

    # 2. Generate plots in memory
    # Decomposition Plot
    if simulator.full_decomposition_result:
        fig_decomp, _ = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        plot_decomposition(
            simulator.full_decomposition_result, sensor_type=sensor_name, fig=fig_decomp
        )
        b64_decomposition = fig_to_base64(fig_decomp)
    else:
        b64_decomposition = ""

    # Comparison Plot
    original_series = simulator.original_data
    fig_comp, _ = plt.subplots(3, 1, figsize=(12, 18))
    plot_comparison(
        original_series, simulated_series, sensor_type=sensor_name, fig=fig_comp
    )
    b64_comparison = fig_to_base64(fig_comp)

    print("Simulation and plot generation complete.")
    return {
        "simulation_data": json_data,
        "plots": {"decomposition": b64_decomposition, "comparison": b64_comparison},
    }
