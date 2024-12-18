import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import numpy as np

class PlotCSV:
    def __init__(self, file_path):
        """
        Initializes the PlotCSV class by loading the CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
        """
        self.df = pd.read_csv(file_path)

    def acc_to_error(self, acc, type='fraction'):
        """
        Converts accuracy to error.
        
        Args:
            acc (float): Accuracy value.
        
        Returns:
            float: Error value.
        """

        if type == 'percentage':
            return 100 - acc
        elif type == 'fraction':
            return 1 - (acc/100)
        elif type == 'logit':
            return 1/(1-acc)
        elif type == 'log':
            return 1/acc
        elif type == 'loss':
            return acc / 100
        
    def acc_to_error_all(self, acc_column, type='fraction'):
        """
        Converts all accuracy values in a column to error values.
        
        Args:
            acc_column (str): Column name containing accuracy values.
        
        Returns:
            pd.Series: Series containing error values.
        """
        return self.df[acc_column].apply(lambda acc: self.acc_to_error(acc, type))
    
    def plot_matplotlib(self, x_column, y_column):
        """
        Plots the data using matplotlib.
        
        Args:
            x_column (str): The column name for the x-axis.
            y_column (str): The column name for the y-axis.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.df[x_column], self.df[y_column], marker='o', linestyle='-', color='b')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'{y_column} vs {x_column} (Matplotlib)')
        plt.grid(True)
        plt.show()
    
    def plot_seaborn(self, x_column, y_column):
        """
        Plots the data using seaborn.
        
        Args:
            x_column (str): The column name for the x-axis.
            y_column (str): The column name for the y-axis.
        """
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.df, x=x_column, y=y_column, marker='o')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'{y_column} vs {x_column} (Seaborn)')
        plt.grid(True)
        plt.show()
        return 
    
    def available_columns(self):
        """ 
        Returns the available columns in the dataset. 
        """
        return self.df.columns.tolist()

    def split_columns_as_dict(self):
        """
        Splits the DataFrame into a dictionary where each column name is a key,
        and the values are lists of the corresponding column values.
        
        Returns:
            dict: A dictionary containing columns and their corresponding values.
        """
        data_dict = self.df.to_dict(orient='list')
        data_dict['Row Index'] = self.df.index.tolist()  # Include row indices
        return data_dict
    
    def split_by_experiment(self, split_column="Dataset Fraction"):
        """
        Splits the DataFrame into subsets based on unique values in the specified column.
        
        Args:
            split_column (str): The column by which to split the DataFrame.
        
        Returns:
            dict: A dictionary where keys are unique values from the split_column, 
                  and values are DataFrames corresponding to each subset.
        """
        if split_column not in self.df.columns:
            raise ValueError(f"Column '{split_column}' not found in the DataFrame.")
        
        # Grouping by the unique values in the specified column
        grouped_data = {key: subset for key, subset in self.df.groupby(split_column)}
        return grouped_data
    
    def plot_mean_by_epoch_range(self, epoch_column, start_epoch, end_epoch):
        """
        Plots the mean values of each column over a specified epoch range.
        
        Args:
            epoch_column (str): The name of the column representing epochs.
            start_epoch (int): The starting epoch value.
            end_epoch (int): The ending epoch value.
        """
        # Filter the DataFrame for the specified epoch range
        subset = self.df[(self.df[epoch_column] >= start_epoch) & (self.df[epoch_column] <= end_epoch)]
        
        # Calculate the mean for each column
        mean_values = subset.mean(numeric_only=True)
        
        # Plotting each column's mean value
        plt.figure(figsize=(12, 6))
        mean_values.plot(kind='bar', color='skyblue')
        plt.title(f"Mean Values of Columns from Epoch {start_epoch} to {end_epoch}")
        plt.ylabel("Mean Value")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_all_column_combinations(self, epoch_column, start_epoch, end_epoch):
        """
        Plots all combinations of numeric columns over a specified epoch range.
        
        Args:
            epoch_column (str): The name of the column representing epochs.
            start_epoch (int): The starting epoch value.
            end_epoch (int): The ending epoch value.
        """
        # Filter the DataFrame based on the specified epoch range
        subset = self.df[(self.df[epoch_column] >= start_epoch) & (self.df[epoch_column] <= end_epoch)]
        #print shape of subset
        #print(subset.shape)

        #take the main of the subset of the epochs using numpy
        



        

        #print(subset.shape)
        
        
        numeric_cols = subset.select_dtypes(include='number').columns
        for x_col in numeric_cols:
            for y_col in numeric_cols:
                if x_col != y_col:
                    #take mean over the epochs range
                    #subset[x_col] = subset[x_col].mean()
                    plt.figure(figsize=(10, 6))
                    plt.plot(subset[x_col], subset[y_col], marker='o')
                    plt.xlabel(x_col.replace('_', ' ').title())
                    plt.ylabel(y_col.replace('_', ' ').title())
                    plt.title(f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}')
                    plt.grid(True)
                    plt.show()
        
    def save_plots_to_folder(self, epoch_column, start_epoch, end_epoch, output_folder='plots'):
        """
        Saves all plots from plot_all_column_combinations to a specified folder.
        
        Args:
            epoch_column (str): The name of the column representing epochs.
            start_epoch (int): The starting epoch value.
            end_epoch (int): The ending epoch value.
            output_folder (str): The folder where plots will be saved.
        """
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Filter the DataFrame based on the specified epoch range
        subset = self.df[(self.df[epoch_column] >= start_epoch) & (self.df[epoch_column] <= end_epoch)]
        numeric_cols = subset.select_dtypes(include='number').columns
        
        for x_col in numeric_cols:
            for y_col in numeric_cols:
                if x_col != y_col:
                    plt.figure(figsize=(10, 6))
                    plt.plot(subset[x_col], subset[y_col], marker='o')
                    plt.xlabel(x_col.replace('_', ' ').title())
                    plt.ylabel(y_col.replace('_', ' ').title())
                    plt.title(f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}')
                    plt.grid(True)
                    
                    # Save plot with descriptive filename
                    filename = f'{y_col}_vs_{x_col}_epoch_{start_epoch}_to_{end_epoch}.png'
                    plt.savefig(os.path.join(output_folder, filename))
                    plt.close()  # Close the plot to avoid displaying it
    
    
    def model_size_plot(self,save=False):
        experiments = self.split_by_experiment()
        for key, df_subset in experiments.items():
            print(f"Experiment for {key}:")
            print(df_subset.head(), "\n")

            # Plot using matplotlib
            #Take only the last epoch
            df_subset = df_subset[df_subset['Epoch'] == df_subset['Epoch'].max()]
            
            #convert accuracy to error
            df_subset['Train Error'] = plotter.acc_to_error_all('Train Accuracy', type='fraction')
            df_subset['Test Error'] = plotter.acc_to_error_all('Test Accuracy', type='fraction')

            #Make a new plot using matplotlib
            plt.figure(figsize=(10, 6))
            plt.plot(df_subset['Model Size'], df_subset['Train Error'], marker='o', linestyle='-', color='b', label='Train Error')
            plt.plot(df_subset['Model Size'], df_subset['Test Error'], marker='o', linestyle='-', color='r', label='Test Error')
            plt.xlabel('Model Size')
            plt.ylabel('Error')
            plt.title(f'Train and Test Error vs Model Size for {key}')
            plt.legend()
            plt.grid(True)
            plt.show()
            #save the plot
            if save:
                plt.savefig('model_size_Error_{key}.png')

            #convert accuracy to error
            df_subset['Train Loss 1'] = plotter.acc_to_error_all('Train Loss', type='loss')
            df_subset['Test Loss 1'] = plotter.acc_to_error_all('Test Loss', type='loss')

            #Make a new plot using matplotlib
            plt.figure(figsize=(10, 6))
            plt.plot(df_subset['Model Size'], df_subset['Train Loss 1'], marker='o', linestyle='-', color='b', label='Train Error')
            plt.plot(df_subset['Model Size'], df_subset['Test Loss 1'], marker='o', linestyle='-', color='r', label='Test Error')
            plt.xlabel('Model Size')
            plt.ylabel('Loss,normalized')
            plt.title(f'Train and Test Loss normalized vs Model Size for {key}')
            plt.legend()
            plt.grid(True)
            plt.show()
            #save the plot
            if save:
                plt.savefig('model_size_Loss_Norm_{key}.png')

            
            #Make a new plot using matplotlib
            plt.figure(figsize=(10, 6))
            plt.plot(df_subset['Model Size'], df_subset['Train Loss'], marker='o', linestyle='-', color='b', label='Train Error')
            plt.plot(df_subset['Model Size'], df_subset['Test Loss'], marker='o', linestyle='-', color='r', label='Test Error')
            plt.xlabel('Model Size')
            plt.ylabel('Loss')
            plt.title(f'Train and Test Loss vs Model Size for {key}')
            plt.legend()
            plt.grid(True)
            plt.show()
            #save the plot
            if save:
                plt.savefig('model_size_Loss_{key}.png')

            #Make a new plot using matplotlib with log scale
            plt.figure(figsize=(10, 6))
            plt.plot(df_subset['Model Size'], df_subset['Train Loss'], marker='o', linestyle='-', color='b', label='Train Error')
            plt.plot(df_subset['Model Size'], df_subset['Test Loss'], marker='o', linestyle='-', color='r', label='Test Error')
            plt.xlabel('Model Size')
            plt.ylabel('Loss')
            plt.title(f'Train and Test Loss vs Model Size for {key}')
            plt.legend()
            plt.grid(True)
            plt.yscale('log')
            plt.show()
            #save the plot
            if save:
                plt.savefig('model_size_Loss_log_{key}.png')
        
    def epoch_plot(self,save=False):
        experiments = self.split_by_experiment()
        for key, df_subset in experiments.items():
            print(f"Experiment for {key}:")
            print(df_subset.head(), "\n")


            df_subset = df_subset[df_subset['Model Size'] == df_subset['Model Size'].max()]
            #convert accuracy to error
            df_subset['Train Error'] = plotter.acc_to_error_all('Train Accuracy', type='fraction')
            df_subset['Test Error'] = plotter.acc_to_error_all('Test Accuracy', type='fraction')

            #Make a new plot using matplotlib
            plt.figure(figsize=(10, 6))
            plt.plot(df_subset['Epoch'], df_subset['Train Error'], marker='o', linestyle='-', color='b', label='Train Error')
            plt.plot(df_subset['Epoch'], df_subset['Test Error'], marker='o', linestyle='-', color='r', label='Test Error')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.title(f'Train and Test Error vs Epoch for {key}')
            plt.legend()
            plt.grid(True)
            plt.show()
            if save:
                plt.savefig('epoch_Error_{key}.png')

            #convert accuracy to error
            df_subset['Train Loss 1'] = plotter.acc_to_error_all('Train Loss', type='loss')
            df_subset['Test Loss 1'] = plotter.acc_to_error_all('Test Loss', type='loss')

            #Make a new plot using matplotlib
            plt.figure(figsize=(10, 6))
            plt.plot(df_subset['Epoch'], df_subset['Train Loss 1'], marker='o', linestyle='-', color='b', label='Train Error')
            plt.plot(df_subset['Epoch'], df_subset['Test Loss 1'], marker='o', linestyle='-', color='r', label='Test Error')
            plt.xlabel('Epoch')
            plt.ylabel('Loss,normalized')
            plt.title(f'Train and Test Loss normalized vs Epoch for {key}')
            plt.legend()
            plt.grid(True)
            plt.show()
            if save:
                plt.savefig('epoch_Error_Norm_{key}.png')

            
            #Make a new plot using matplotlib
            plt.figure(figsize=(10, 6))
            plt.plot(df_subset['Epoch'], df_subset['Train Loss'], marker='o', linestyle='-', color='b', label='Train Error')
            plt.plot(df_subset['Epoch'], df_subset['Test Loss'], marker='o', linestyle='-', color='r', label='Test Error')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Train and Test Loss vs Epoch for {key}')
            plt.legend()
            plt.grid(True)
            plt.show()
            if save:
                plt.savefig('epoch_Loss_{key}.png')

            #Make a new plot using matplotlib with log scale
            plt.figure(figsize=(10, 6))
            plt.plot(df_subset['Epoch'], df_subset['Train Loss'], marker='o', linestyle='-', color='b', label='Train Error')
            plt.plot(df_subset['Epoch'], df_subset['Test Loss'], marker='o', linestyle='-', color='r', label='Test Error')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Train and Test Loss vs Epoch for {key}')
            plt.legend()
            plt.grid(True)
            plt.yscale('log')
            plt.show()
            #save the plot
            if save:
                plt.savefig('epoch_Loss_log_{key}.png')
    
    def heatmap_plot(self,save=False):

        experiments = self.split_by_experiment()
        for key, df_subset in experiments.items():
            print(f"Experiment for {key}:")
            print(df_subset.head(), "\n")

            #Now we need to make a 2D plot of epoch and modelsize and loss/error as heatmap
            #Using matplotlib
            # Extract the necessary columns from the DataFrame

            epoch_column = 'Epoch'
            model_size_column = 'Model Size'
            loss_column = 'Test Error'

            #df_subset = df_subset[df_subset['Model Size'] == df_subset['Model Size'].max()]
            #df_subset = df_subset[df_subset['Epoch'] == df_subset['Epoch'].max()]
            #convert accuracy to error
            df_subset['Test Error'] = plotter.acc_to_error_all('Test Accuracy', type='fraction')
            epochs = df_subset[epoch_column]
            model_sizes = df_subset[model_size_column]
            losses = df_subset[loss_column]
            print(epoch_column)

            
            #data = df_subset.pivot(index=epochs, columns=model_sizes, values=losses)     
            data = df_subset.pivot(index=epoch_column, columns=model_size_column, values=loss_column)
            data = data.iloc[::-1]




            #print(data)
            # Create a 2D grid of epoch and model size values
            #epoch_grid, model_size_grid = np.meshgrid(epochs, model_sizes)

            # Reshape the losses into a 2D array
            #oss_grid = np.reshape(losses, (len(model_sizes), len(epochs)))

            
        
        # Plot using matplotlib
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(data, cmap='viridis', cbar=True)
            plt.title('title')
            plt.xlabel('xlabel')
            plt.ylabel('ylabel')
            plt.show()

            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(data, cmap='viridis', cbar=True)
            plt.title('title')
            plt.xlabel('xlabel')
            plt.ylabel('ylabel')
            #log scale
            plt.yscale('log')

            
            # Manually create log-like tick positions and labels
            tick_positions = [0, np.log10(10), np.log10(100), np.log10(400)]
            tick_labels = ['1', '10', '100', '400']

            # Scale y-ticks for log-like axis
            ax.set_yticks(tick_positions)  # Approximate positions for the log effect
            ax.set_yticklabels(tick_labels)
            plt.show()
            #save the plot
            if save:
                plt.savefig('heatmap_{key}.png')
    
# The class is now ready. It can be instantiated and used to plot data once the CSV path is provided.

if __name__=="__main__":

    path = "./Deep_learning/13-11-2024/DL_training_log_merged.csv"
    # Initialize the class with the CSV file path
    plotter = PlotCSV(path)

    # Print available columns
    print("Available columns:", plotter.available_columns())

    # Split the DataFrame into a dictionary
    #data_dict = plotter.split_columns_as_dict()
    #print(data_dict)

    experiments = plotter.split_by_experiment()

    # Print each experiment's first few rows
    
    plotter.model_size_plot()
    #plotter.epoch_plot()
    #plotter.heatmap_plot()

    # Plot all combinations for epochs between 370 and 400
    #plotter.plot_all_column_combinations(epoch_column="Epoch", start_epoch=400, end_epoch=400)

    # Save all plots to the 'plots' folder
    #plotter.save_plots_to_folder(epoch_column="Epoch", start_epoch=370, end_epoch=400, output_folder="output_plots")
    
    #Plot the mean of all columns for epochs between 370 and 400
    #plotter.plot_mean_by_epoch_range(epoch_column="Epoch", start_epoch=370, end_epoch=400)

    # Plot using matplotlib
    #plotter.plot_matplotlib(x_column='Epoch', y_column='Train Loss')

    # Plot using seaborn
    #plotter.plot_seaborn(x_column='Epoch', y_column='Test Accuracy')

