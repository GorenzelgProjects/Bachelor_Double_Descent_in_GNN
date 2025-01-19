import pandas as pd
import numpy as np
import pickle

# Load the CSV file
data = pd.read_csv('./concatenated_1.csv')

 #Define dataset fractions
fractions = data['Dataset Fraction'].unique()

# Process the data for each fraction
output_files = {}
for fraction in fractions:
    # Filter data by fraction
    filtered_data = data[data['Dataset Fraction'] == fraction]

    # Initialize the output list for this fraction
    Mlist_format = []

    # Group by 'Model Size'
    for model_size, group in filtered_data.groupby('Model Size'):
        # Ensure no duplicate entries for 'Epoch'
        group = group.drop_duplicates(subset=['Epoch'])

        # Create arrays for metrics
        test_error = group.sort_values('Epoch')['Test Error'].to_numpy()
        test_loss = group.sort_values('Epoch')['Test Loss'].to_numpy()
        train_error = group.sort_values('Epoch')['Train Error'].to_numpy()
        train_loss = group.sort_values('Epoch')['Train Loss'].to_numpy()

        # Create dictionary for this model size
        entry = {
            'Test Error': test_error,
            'Test Loss': test_loss,
            'Train Error': train_error,
            'Train Loss': train_loss,
        }
        Mlist_format.append(entry)

    # Save the formatted data to the output dictionary
    output_files[f"Mlist_fraction_{fraction}.pkl"] = Mlist_format

# Save the data as pickle files
for file_name, content in output_files.items():
    with open(file_name, 'wb') as f:
        pickle.dump(content, f)

print("Files generated:", list(output_files.keys()))