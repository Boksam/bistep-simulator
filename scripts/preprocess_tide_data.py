"""Preprocesses raw tide data into a clean, resampled format.

This script reads raw tide observation data from Excel files, processes it,
calculates averages over a specified time period (e.g., hourly), and saves
the result as a CSV file.
"""

import glob
import os

import numpy as np
import pandas as pd

# --- Configuration ---
# Note: Assumes the script is run from the project root directory.
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "backend", "data", "raw", "tide")
OUTPUT_DIR = os.path.join(ROOT_DIR, "backend", "data", "processed")

# Resampling frequency: 'H' (hourly), 'D' (daily), 'T' (minute)
RESAMPLE_FREQ = "H"

FILE_PATTERN = "tide_202*.xlsx"
DATETIME_COL = "관측일시"
TEMPERATURE_COL = "수온"
SALINITY_COL = "염분"
TIDAL_LEVEL_COL = "조위"


def find_file_paths(directory: str, pattern: str) -> list[str]:
    """Finds file paths matching a pattern in a directory.

    Args:
        directory: The directory to search.
        pattern: The glob pattern to match.

    Returns:
        A sorted list of file paths.

    Raises:
        FileNotFoundError: If no files match the pattern.
    """
    file_pattern = os.path.join(directory, pattern)
    file_paths = sorted(glob.glob(file_pattern))
    if not file_paths:
        raise FileNotFoundError(
            f"No files found in '{directory}' matching '{pattern}'."
        )
    print(f"Found {len(file_paths)} files to process.")
    return file_paths


def load_and_combine_data(
    file_paths: list[str],
    target_cols: list[str]
) -> pd.DataFrame:
    """Loads and combines data from multiple Excel files.

    Args:
        file_paths: A list of paths to the Excel files.
        target_cols: A list of column names to extract, besides the datetime column.

    Returns:
        A single DataFrame containing the combined data.
    """
    all_dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path, engine="openpyxl")

            cols_to_extract = [DATETIME_COL]
            for col in target_cols:
                if col in df.columns:
                    cols_to_extract.append(col)

            if len(cols_to_extract) > 1:  # Has datetime and at least one data column
                all_dfs.append(df[cols_to_extract])
            else:
                print(
                    f"Warning: Required columns not found in "
                    f"{os.path.basename(file_path)}. Skipping."
                )
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")

    if not all_dfs:
        print("No data could be loaded.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def preprocess_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Preprocesses the combined DataFrame for a specific target column.

    This includes converting data types, handling missing values, and setting
    a DatetimeIndex.

    Args:
        df: The raw DataFrame to preprocess.
        target_col: The name of the target column to process.

    Returns:
        The preprocessed DataFrame.
    """
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df.dropna(subset=[DATETIME_COL, target_col], inplace=True)
    df.set_index(DATETIME_COL, inplace=True)
    df.sort_index(inplace=True)
    return df


def resample_and_average(df: pd.DataFrame, freq: str, target_col: str) -> pd.Series:
    """Resamples the data to a specified frequency and calculates the mean.

    Args:
        df: The DataFrame to resample (must have a DatetimeIndex).
        freq: The resampling frequency (e.g., 'H', 'D').
        target_col: The name of the target column to resample.

    Returns:
        A Series with the resampled average value.
    """
    return df[target_col].resample(freq).mean()


def save_data_to_csv(data: pd.Series, output_path: str, output_col_name: str):
    """Saves the processed data to a CSV file.

    Args:
        data: The data to save.
        output_path: The path to the output CSV file.
        output_col_name: The name for the data column in the CSV file.
    """
    df_to_save = data.reset_index()
    df_to_save.columns = ["timestamp", output_col_name]
    # Replace zero values with NaN as they might be invalid readings
    df_to_save.loc[df_to_save[output_col_name] == 0.0, output_col_name] = np.nan
    df_to_save.to_csv(output_path, header=True, index=False)
    print(f"\nSuccessfully saved processed data to:\n{output_path}")


def main():
    """Main function to run the data processing pipeline for multiple variables."""
    print("--- Starting Tide Data Preprocessing ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    variables_to_process = {
        "temperature": {"col": TEMPERATURE_COL, "output_name": "temperature"},
        "salinity": {"col": SALINITY_COL, "output_name": "salinity"},
        "tidal_level": {"col": TIDAL_LEVEL_COL, "output_name": "tidal_level"},
    }

    try:
        file_paths = find_file_paths(DATA_DIR, FILE_PATTERN)

        target_cols = [info["col"] for info in variables_to_process.values()]
        combined_df = load_and_combine_data(file_paths, target_cols)

        if combined_df.empty:
            print("Exiting: No data was loaded.")
            return

        for var_name, var_info in variables_to_process.items():
            col_name = var_info["col"]
            output_col_name = var_info["output_name"]

            if col_name not in combined_df.columns:
                print(
                    f"Skipping '{var_name}' processing: Column '{col_name}' not found in the loaded data."
                )
                continue

            print(f"\n--- Processing {var_name.capitalize()} ---")

            # Process each variable from a clean copy of the relevant columns
            var_df = combined_df[[DATETIME_COL, col_name]].copy()

            processed_df = preprocess_data(var_df, col_name)
            resampled_data = resample_and_average(
                processed_df, RESAMPLE_FREQ, col_name
            )

            freq_map = {"D": "daily", "H": "hourly", "T": "minute", "S": "second"}
            output_filename = f"{freq_map.get(RESAMPLE_FREQ, 'resampled')}_avg_water_{output_col_name}.csv"
            output_csv_path = os.path.join(OUTPUT_DIR, output_filename)

            save_data_to_csv(resampled_data, output_csv_path, output_col_name)

            print(f"\n--- {var_name.capitalize()} Data Preview ---")
            print(resampled_data.head())

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("\n--- Preprocessing Finished ---")


if __name__ == "__main__":
    main()