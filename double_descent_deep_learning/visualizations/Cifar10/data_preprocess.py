import pandas as pd
import numpy as np
import os

def preprocess_training_logs(folder_paths):
    """
    Processes training log files from multiple folders to compute the mean and standard deviation
    of specified metrics for each model size and data fraction.

    Args:
        folder_paths (list of str): List of folder paths, each containing the training log .csv files for a run.

    Returns:
        list of pd.DataFrame: A list of DataFrames (one per folder) containing the stacked data for each run.
    """
    all_runs_data = []

    # Iterate through each folder
    for folder_path in folder_paths:
        # Gather all .csv files from the folder
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

        # Ensure files are sorted to maintain consistent order
        file_paths = sorted(file_paths)

        # Stack files within the folder
        stacked_data = pd.concat([pd.read_csv(path) for path in file_paths])
        all_runs_data.append(stacked_data)

    return all_runs_data

def calculate_metrics(runs_data):
    """
    Calculates mean and standard deviation metrics from stacked run data.

    Args:
        runs_data (list of pd.DataFrame): List of DataFrames, each representing stacked data for a run.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated mean and standard deviation metrics across all runs.
    """
    # Concatenate all runs into a single DataFrame
    combined_data = pd.concat(runs_data)

    results = []

    # Group by model size and dataset fraction
    grouped = combined_data.groupby(["Model Size", "Dataset Fraction"])

    for (model_size, data_fraction), group in grouped:
        # Filter the last 10 epochs
        last_10_epochs = group[group["Epoch"] > group["Epoch"].max() - 10]

        # Compute metrics for the last 10 epochs
        test_error = 1 - (last_10_epochs["Test Accuracy"] / 100)
        mse_loss = last_10_epochs["Test Loss"]
        mse_train_loss = last_10_epochs["Train Loss"]

        # Aggregate metrics: mean and standard deviation
        metrics = {
            "Model Size": model_size,
            "Dataset Fraction": data_fraction,
            "Mean Test Error": test_error.mean(),
            "Std Test Error": test_error.std(),
            "Mean MSE Loss": mse_loss.mean(),
            "Std MSE Loss": mse_loss.std(),
            "Mean MSE Train Loss": mse_train_loss.mean(),
            "Std MSE Train Loss": mse_train_loss.std(),
        }

        results.append(metrics)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def save_metrics_by_fraction(metrics_df, output_folder):
    """
    Saves the metrics into separate CSV files, one for each data fraction.

    Args:
        metrics_df (pd.DataFrame): The DataFrame containing metrics for all data fractions.
        output_folder (str): The folder to save the output CSV files.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Group by dataset fraction and save each group to a separate CSV file
    for data_fraction, group in metrics_df.groupby("Dataset Fraction"):
        file_name = f"metrics_fraction_{data_fraction:.2f}.csv"
        output_path = os.path.join(output_folder, file_name)
        group.to_csv(output_path, index=False)

# folder_paths = ["/path/to/run1", "/path/to/run2", "/path/to/run3"]
# runs_data = preprocess_training_logs(folder_paths)
# metrics_results = calculate_metrics(runs_data)
# save_metrics_by_fraction(metrics_results, "/path/to/output")
# Example usage:
folder_paths = ["./Run1", "./Run2", "./Run3"]
runs_data = preprocess_training_logs(folder_paths)
metrics_results = calculate_metrics(runs_data)
save_metrics_by_fraction(metrics_results, "./output")

