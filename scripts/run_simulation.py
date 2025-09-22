"""Main entry point for the BISTEP data simulation project.

This script coordinates the analysis of historical data and the generation
of simulated data.
"""

import os

from simulator.simulators.water_temperature import WaterTemperatureSimulator
from simulator.utils import plotting

# --- Configuration ---
ROOT_DIR = os.getcwd()
PROCESSED_DATA_PATH = os.path.join(
    ROOT_DIR, "data", "processed", "hourly_avg_water_temperature.csv"
)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports", "figures")


def main():
    """Main function to run the simulation and generate outputs."""
    print("--- Starting Water Temperature Simulation ---")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    try:
        # 1. Analyze historical data
        simulator = WaterTemperatureSimulator()
        simulator.run_analysis(PROCESSED_DATA_PATH, trend_degree=2)

        # 2. Plot decomposition results
        if simulator.full_decomposition_result:
            plotting.plot_decomposition(
                simulator.full_decomposition_result, REPORTS_DIR
            )

        # 3. Simulate new data for the same period as the original data
        if simulator.original_data is not None:
            start_date = simulator.original_data.index.min().strftime("%Y-%m-%d")
            end_date = simulator.original_data.index.max().strftime("%Y-%m-%d")

            print(f"\nSimulating data from {start_date} to {end_date}...")
            simulated_data = simulator.simulate(start_date, end_date, seed=42)

            print("\n--- Simulation vs. Original Data ---")
            print(f"Simulated Mean: {simulated_data.mean():.2f}°C")
            print(f"Original Mean:  {simulator.original_data.mean():.2f}°C")
            print(f"Simulated Std:  {simulated_data.std():.2f}°C")
            print(f"Original Std:   {simulator.original_data.std():.2f}°C")

            # 4. Save simulated data
            output_excel_path = os.path.join(
                REPORTS_DIR, "simulated_water_temperature.xlsx"
            )
            simulated_data.to_excel(output_excel_path)
            print(f"\nSimulated data saved to: {output_excel_path}")

            # 5. Plot comparison
            plotting.plot_comparison(
                simulator.original_data, simulated_data, REPORTS_DIR
            )

    except FileNotFoundError as e:
        print(f"Error: Could not find data file at {PROCESSED_DATA_PATH}")
        print("Please run the preprocessing script first:")
        print("  python scripts/preprocess_tide_data.py")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("\n--- Simulation Finished ---")


if __name__ == "__main__":
    main()
