import matplotlib.pyplot as plt
import pandas as pd

def load_merge_and_sort_csv(file_paths):
    """
    Load multiple CSV files, merge them, and sort by Depth (layers), 
    Width (hidden_channels), and Epochs.
    
    Parameters:
        file_paths (list of str): List of paths to the CSV files.
    
    Returns:
        pd.DataFrame: Sorted DataFrame containing all merged data.
    """
    dataframes = [pd.read_csv(file) for file in file_paths]
    merged_data = pd.concat(dataframes, ignore_index=True)
    
    # Sort by layers (Depth), hidden_channels (Width), and epochs
    sorted_data = merged_data.sort_values(by=["layers", "hidden_channels", "epochs"]).reset_index(drop=True)
    return sorted_data

def plot_with_epoch_filter(data, x_param, y_param, epoch_range=None, specific_epoch=None, 
                           title="Plot", xlabel=None, ylabel=None):
    """
    Plot data filtered by a range of epochs or a specific epoch.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data to plot.
        x_param (str): Column name for the x-axis.
        y_param (str): Column name for the y-axis.
        epoch_range (tuple, optional): Range of epochs (start, end) for averaging.
        specific_epoch (int, optional): Specific epoch to filter and plot.
        title (str, optional): Title of the plot. Defaults to "Plot".
        xlabel (str, optional): Label for the x-axis. Defaults to None.
        ylabel (str, optional): Label for the y-axis. Defaults to None.
    
    Returns:
        None
    """
    if epoch_range:
        start, end = epoch_range
        filtered_data = data[(data['epochs'] >= start) & (data['epochs'] <= end)]
        # Calculate mean and std
        aggregated = filtered_data.groupby(x_param)[y_param].agg(['mean', 'std']).reset_index()
        x = aggregated[x_param]
        y_mean = aggregated['mean']
        y_std = aggregated['std']

        plt.figure(figsize=(10, 6))
        plt.errorbar(x, y_mean, yerr=y_std, fmt='o-', capsize=5, label=f"{y_param} (Mean ± Std)")
        plt.title(title)
        plt.xlabel(xlabel if xlabel else x_param)
        plt.ylabel(ylabel if ylabel else y_param)
        plt.legend()
        plt.grid(True)
        plt.show()

    elif specific_epoch is not None:
        filtered_data = data[data['epochs'] == specific_epoch]

        plt.figure(figsize=(10, 6))
        plt.plot(filtered_data[x_param], filtered_data[y_param], 'o-', label=f"Epoch {specific_epoch}")
        plt.title(title)
        plt.xlabel(xlabel if xlabel else x_param)
        plt.ylabel(ylabel if ylabel else y_param)
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Please specify either an epoch range or a specific epoch.")


if __name__ == "__main__":
    file_path = ["./logs/26-11-2024/gnn_training_log_1.csv","./logs/26-11-2024/gnn_training_log_2.csv","./logs/26-11-2024/gnn_training_log_3.csv"]
    merged_data = load_merge_and_sort_csv(file_path)
# Example usage for an epoch range
    plot_with_epoch_filter(merged_data, x_param="hidden_channels", y_param="mad_value", 
                        epoch_range=(3000, 4000), title="MAD Value vs Hidden Channels (Epochs 10-50)", 
                        xlabel="Hidden Channels", ylabel="MAD Value")

    # Example usage for a specific epoch
    plot_with_epoch_filter(merged_data, x_param="hidden_channels", y_param="mad_value", 
                        specific_epoch=3990, title="MAD Value vs Hidden Channels (Epoch 50)", 
                        xlabel="Hidden Channels", ylabel="MAD Value")

    #Example usage for an epoch range and depth vs mad value
    plot_with_epoch_filter(merged_data, x_param="layers", y_param="mad_value", 
                        epoch_range=(3000, 4000), title="MAD Value vs Layers (Epochs 10-50)", 
                        xlabel="Layers", ylabel="MAD Value")
    
    #Example usage for an epoch range and depth vs mad value
    plot_with_epoch_filter(merged_data, x_param="layers", y_param="mad_value", 
                        specific_epoch=3990, title="MAD Value vs Layers (Epoch 50)", 
                        xlabel="Layers", ylabel="MAD Value")