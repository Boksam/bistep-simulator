"""
FastAPI server for providing simulated sensor data.
"""

import base64
import io
import os
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pandas import Timedelta
from plotting import plot_comparison, plot_decomposition
from pydantic import BaseModel, Field
from salinity import SalinitySimulator
from tidal_level import TidalLevelSimulator
from water_temperature import WaterTemperatureSimulator
from anode_lifetime import simulate_anode_lifetime

# --- Configuration Loading ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Warning: '{CONFIG_PATH}' not found. Using default configurations.")
    config = {}
except yaml.YAMLError as e:
    print(
        f"Warning: Error parsing '{CONFIG_PATH}': {e}. Using default configurations."
    )
    config = {}

PROCESSED_WATER_TEMP_PATH = os.path.join(os.path.dirname(__file__), "data",
                                         "processed",
                                         "hourly_avg_water_temperature.csv")
PROCESSED_SALINITY_PATH = os.path.join(os.path.dirname(__file__), "data",
                                       "processed",
                                       "hourly_avg_water_salinity.csv")
PROCESSED_TIDAL_LEVEL_PATH = os.path.join(os.path.dirname(__file__), "data",
                                          "processed",
                                          "hourly_avg_water_tidal_level.csv")

# Create a FastAPI app instance
app = FastAPI(
    title="Sensor Simulation API",
    description="Provides simulated time-series data for various sensors.",
    version="1.0.0",
)

# --- CORS Middleware ---
origins = [
    "http://localhost:5173",
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
    interval: Optional[str] = Field(
        "1h",
        description=
        "Data aggregation interval (e.g., '1h', '12h', '1d', '7d').",
    )


class PlotResponse(BaseModel):
    """Response model for plots."""

    decomposition: str
    comparison: str


class SimulationResponse(BaseModel):
    """Response model for simulation data and plots."""

    simulation_data: List[Dict]
    plots: PlotResponse


# --- Application State ---
simulators: Dict[str, WaterTemperatureSimulator | SalinitySimulator
                 | TidalLevelSimulator] = {}


# --- Helper Functions ---
def fig_to_base64(fig) -> str:
    """Converts a Matplotlib figure to a Base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(
        buf.getvalue()).decode("utf-8")


def parse_interval(interval_str: str) -> Timedelta:
    """Parses a human-readable interval string into a pandas Timedelta object."""
    try:
        return pd.to_timedelta(interval_str)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=
            f"Invalid interval format: '{interval_str}'. Use formats like '1h', '12h', '1d'.",
        )


# --- Events ---
@app.on_event("startup")
async def startup_event():
    """Loads and trains simulation models to avoid re-training on every API call."""
    print("Server is starting up...")

    # Load and train water temperature simulator
    print("Loading and training water temperature simulator...")
    try:
        water_temp_simulator = WaterTemperatureSimulator()
        water_temp_simulator.run_analysis(PROCESSED_WATER_TEMP_PATH,
                                          trend_degree=1)
        simulators["water_temperature"] = water_temp_simulator
        print("Water temperature simulator loaded successfully.")
    except FileNotFoundError:
        print(
            f"ERROR: Could not load water temperature simulator. Data file not found at {PROCESSED_WATER_TEMP_PATH}."
        )
    except Exception as e:
        print(
            f"An unexpected error occurred during water temperature simulator loading: {e}"
        )

    # Load and train salinity simulator
    print("\nLoading and training salinity simulator...")
    try:
        salinity_simulator = SalinitySimulator()
        salinity_simulator.run_analysis(PROCESSED_SALINITY_PATH,
                                        trend_degree=0)
        simulators["salinity"] = salinity_simulator
        print("Salinity simulator loaded successfully.")
    except FileNotFoundError:
        print(
            f"ERROR: Could not load salinity simulator. Data file not found at {PROCESSED_SALINITY_PATH}."
        )
    except Exception as e:
        print(
            f"An unexpected error occurred during salinity simulator loading: {e}"
        )

    # Load and train tidal level simulator
    print("\nLoading and training tidal level simulator...")
    try:
        tidal_level_simulator = TidalLevelSimulator()
        tidal_level_simulator.run_analysis(PROCESSED_TIDAL_LEVEL_PATH,
                                           trend_degree=0)
        simulators["tidal_level"] = tidal_level_simulator
        print("Tidal level simulator loaded successfully.")
    except FileNotFoundError:
        print(
            f"ERROR: Could not load tidal level simulator. Data file not found at {PROCESSED_TIDAL_LEVEL_PATH}."
        )
    except Exception as e:
        print(
            f"An unexpected error occurred during tidal level simulator loading: {e}"
        )


# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint providing basic information about the API."""
    return {
        "message": "Welcome to the Sensor Simulation API!",
        "available_simulators": list(simulators.keys()),
        "documentation": "/docs",
    }


class AnodeLifetimeSimulationRequest(BaseModel):
    """Request model for the anode lifetime simulation endpoint."""

    start_date: str
    end_date: str
    interval: str = Field(
        "1d", description="Data aggregation interval (e.g., '1h', '1d').")
    constant_duration: int = Field(
        24 * 30 * 6,  # 6 months
        description=
        "The number of hours the anode lifetime remains at its maximum value before starting to decay.",
    )
    decay_rate: float = Field(
        0.001,
        description=
        "The hourly rate of exponential decay after the constant duration.",
    )
    noise_level: float = Field(
        0.01,
        description="The standard deviation of the Gaussian noise to add.")


@app.post("/simulations/anode-lifetime")
async def run_anode_lifetime_simulation(
        request: AnodeLifetimeSimulationRequest):
    """Runs a new simulation for anode lifetime and returns the data."""
    print(
        f"Generating anode lifetime simulation with parameters: {request.dict()}"
    )

    try:
        # 1. Generate simulated data
        simulated_df = simulate_anode_lifetime(
            constant_duration=request.constant_duration,
            decay_rate=request.decay_rate,
            noise_level=request.noise_level,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
        )

        # 2. Format for response
        result_df = simulated_df.reset_index()
        result_df.columns = ["timestamp", "value"]
        json_data = result_df.to_dict(orient="records")

        # 3. Define the formula string
        formula = (
            r"L(t) = 1 + \mathcal{N}(0, \sigma^2), \quad \text{for } t < T_c\\"
            r"L(t) = \exp(-k \cdot (t - T_c)) + \mathcal{N}(0, \sigma^2), \quad \text{for } t \geq T_c"
        )

        print("Anode lifetime simulation complete.")
        return {"simulation_data": json_data, "formula": formula}
    except Exception as e:
        print(f"An error occurred during anode lifetime simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulations/{sensor_name}", response_model=SimulationResponse)
async def run_simulation(sensor_name: str, request: SimulationRequest):
    """Runs a new simulation and returns the data along with visualization plots."""
    if sensor_name not in simulators:
        raise HTTPException(
            status_code=404,
            detail=
            f"Simulator for '{sensor_name}' not found or failed to load.",
        )

    try:
        datetime.strptime(request.start_date, "%Y-%m-%d")
        datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Please use YYYY-MM-DD.")

    # Validate and parse the interval
    requested_interval_td = parse_interval(request.interval)

    # Get simulator-specific minimum interval from config, with a fallback
    min_interval_str = (config.get("simulators",
                                   {}).get(sensor_name,
                                           {}).get("min_interval", "1h"))
    min_interval_td = parse_interval(min_interval_str)

    if requested_interval_td < min_interval_td:
        raise HTTPException(
            status_code=400,
            detail=
            f"Interval for '{sensor_name}' cannot be less than its minimum of {min_interval_str}.",
        )

    simulator = simulators[sensor_name]
    print(
        f"Generating simulation for '{sensor_name}' from {request.start_date} to {request.end_date} with interval {request.interval}..."
    )

    # 1. Generate simulated data at the requested interval
    simulated_series = simulator.simulate(
        start_date=request.start_date,
        end_date=request.end_date,
        freq=request.interval,  # Pass the interval string directly
    )

    # 2. Resample original data for the comparison plot
    original_series = simulator.original_data.resample(
        requested_interval_td).mean()

    result_df = simulated_series.reset_index()
    result_df.columns = ["timestamp", "value"]
    json_data = result_df.to_dict(orient="records")

    # 3. Generate plots in memory
    # Decomposition Plot (on original hourly data)
    if simulator.full_decomposition_result:
        fig_decomp, _ = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        plot_decomposition(simulator.full_decomposition_result,
                           sensor_type=sensor_name,
                           fig=fig_decomp)
        b64_decomposition = fig_to_base64(fig_decomp)
    else:
        b64_decomposition = ""

    # Comparison Plot (on resampled data)
    fig_comp, _ = plt.subplots(3, 1, figsize=(12, 18))
    plot_comparison(original_series,
                    simulated_series,
                    sensor_type=sensor_name,
                    fig=fig_comp)
    b64_comparison = fig_to_base64(fig_comp)

    print("Simulation and plot generation complete.")
    return {
        "simulation_data": json_data,
        "plots": {
            "decomposition": b64_decomposition,
            "comparison": b64_comparison
        },
    }
