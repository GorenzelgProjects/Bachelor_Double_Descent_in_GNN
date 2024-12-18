import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        self.results = {}

    def record(self, param_name, param_value, best_train_loss, test_loss):
        """
        Record the best train loss and the test loss for a given parameter configuration.

        Args:
            param_name (str): The name of the parameter being varied (e.g., "hidden_channels", "layers").
            param_value (int or float): The value of the parameter (e.g., hidden channel size).
            best_train_loss (float): The best train loss observed during training.
            test_loss (float): The test loss after training.
        """
        if param_name not in self.results:
            self.results[param_name] = []

        # Append the parameter value, best train loss, and test loss
        self.results[param_name].append((param_value, best_train_loss, test_loss))

    def plot(self):
        """
        Plot the best train loss and test loss for each parameter configuration.
        """
        for param_name, values in self.results.items():
            # Sort the values by the parameter (x-axis)
            values.sort(key=lambda x: x[0])

            # Separate the x values, best train losses, and test losses
            x_values = [v[0] for v in values]
            best_train_losses = [v[1] for v in values]
            test_losses = [v[2] for v in values]

            # Plot the best train loss vs parameter values
            plt.plot(x_values, best_train_losses, marker='o', label=f'{param_name} vs Best Train Loss')

            # Plot the test loss vs parameter values
            plt.plot(x_values, test_losses, marker='x', label=f'{param_name} vs Test Loss')

        plt.xlabel('Parameter Value')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train and Test Loss Performance')
        plt.grid(True)
        plt.show()
