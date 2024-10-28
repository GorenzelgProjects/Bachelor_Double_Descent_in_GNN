import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file generated from the training loop
csv_file_path = './training_metrics.csv'  # Replace with the correct path to the CSV file
df = pd.read_csv(csv_file_path)

# Ensure the CSV columns: ['Model Size', 'Epoch', 'Train Loss', 'Test Loss', 'Train Error', 'Test Error']

# Example plot for Training and Test Loss over Epochs for different model sizes
def plot_loss_for_model_size(df, model_size):
    # Filter the dataframe for the current model size
    df_filtered = df[df['Model Size'] == model_size]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_filtered['Epoch'], df_filtered['Train Loss'], label='Train Loss')
    plt.plot(df_filtered['Epoch'], df_filtered['Test Loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Test Loss for Model Size k={model_size}')
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(f'loss_plot_k_{model_size}.png')
    plt.show()

# Example plot for Training and Test Error over Epochs for different model sizes
def plot_error_for_model_size(df, model_size):
    # Filter the dataframe for the current model size
    df_filtered = df[df['Model Size'] == model_size]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_filtered['Epoch'], df_filtered['Train Error'], label='Train Error')
    plt.plot(df_filtered['Epoch'], df_filtered['Test Error'], label='Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.title(f'Training and Test Error for Model Size k={model_size}')
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(f'error_plot_k_{model_size}.png')
    plt.show()

def extract_final_errors(df):
    final_errors = df.groupby('Model Size').apply(lambda x: x[x['Epoch'] == x['Epoch'].max()]).reset_index(drop=True)
    return final_errors[['Model Size', 'Train Error', 'Test Error']]

# Function to plot the final train and test error vs model size
def plot_final_errors(df):
    final_errors = extract_final_errors(df)
    
    plt.figure(figsize=(10, 6))
    
    # Plot train and test errors
    plt.plot(final_errors['Model Size'], final_errors['Train Error'], label='Final Train Error', marker='o')
    plt.plot(final_errors['Model Size'], final_errors['Test Error'], label='Final Test Error', marker='o')
    
    # Labeling the plot
    plt.xlabel('Model Size (k)')
    plt.ylabel('Error (%)')
    plt.title('Final Train and Test Error vs Model Size')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('final_errors_vs_model_size.png')
    plt.show()



# Loop through different model sizes and generate plots
for model_size in df['Model Size'].unique():
    plot_loss_for_model_size(df, model_size)
    plot_error_for_model_size(df, model_size)

# Call the plotting function
plot_final_errors(df)

# You can further modify this to fit the specific plots and styles already present in your notebook
