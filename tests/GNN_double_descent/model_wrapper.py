import json
import torch
import torch.nn.functional as F
from models.conventional_models import GCN, GAT
from plot import Plotter
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import csv
import os


# Wrapper class to handle different model creation, training, and testing
class ModelWrapper:
    '''
        This class is a wrapper around the PyTorch model classes. It allows for easy model creation, training, and testing.
        The class also includes a hyperparameter search method to loop through different hyperparameter combinations.

        Parameters:
            model_name (str): Name of the model to use. Supported models are 'GAT'.
            num_features (int): Number of input features.
            out_channels (int): Number of output channels.
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Methods:
            build_model(): Builds the model based on the specified model name and keyword arguments.
            get_optimizer(optimizer_name, learning_rate): Returns an optimizer based on the optimizer name and learning rate.
            get_loss_function(loss_name): Returns a loss function based on the loss name.
            train(data, optimizer_name="adam", loss_name="cross_entropy", epochs=100, learning_rate=0.001): Trains the model on the data.
            test(data, loss_name="cross_entropy"): Tests the model on the data.
            hyperparameter_search(data, layer_range, hidden_channels_range, epoch_range, 
                activation_options, optimizer, loss, learning_rate, num_heads): Loops through different hyperparameter combinations and trains the model.

        Example usage:
            model = ModelWrapper(model_name='GAT', num_features=1433, out_channels=7, num_heads=8)
            model.train(data, optimizer_name='adam', loss_name='cross_entropy', epochs=100, learning_rate=0.01)
            model.test(data, loss_name='cross_entropy')
            model.hyperparameter_search(data, layer_range={'min': 1, 'max': 2, 'step': 1}, 
                hidden_channels_range={'min': 8, 'max': 16, 'step': 8}, 
                epoch_range={'min': 100, 'max': 200, 'step': 100}, 
                activation_options=['relu', 'elu'], optimizer='adam', 
                loss='cross_entropy', learning_rate=0.01, num_heads=8)
    '''

    def __init__(self, model_name, num_features, out_channels, hidden_channels, output_path, **kwargs):
        # Dictionary mapping model names to their constructors
        self.model_constructors = {
            'GCN': GCN,
            'GAT': GAT,
        }

        # Initialize Plotter
        self.plotter = Plotter()

        # Store model type
        self.model_name = model_name
        self.num_features = num_features
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.output_path = output_path

        # Store the arguments needed for model initialization
        self.model_kwargs = kwargs

        # Build the model
        self.model = self.build_model()

    def build_model(self):
        # Get the model constructor based on the model name
        model_constructor = self.model_constructors[self.model_name]
        
        # Create an instance of the model by passing the keyword arguments
        model = model_constructor(num_features=self.num_features, hidden_channels=self.hidden_channels, out_channels=self.out_channels, **self.model_kwargs)
        
        return model
    
    def get_optimizer(self, optimizer_name, learning_rate):
        if optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def get_loss_function(self, loss_name):
        if loss_name == "cross_entropy":
            return F.cross_entropy
        elif loss_name == "mse":
            return F.mse_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def train(self, data, optimizer_name="adam", loss_name="cross_entropy", epochs=100, learning_rate=0.001):
        # Get the optimizer and loss function based on names
        optimizer = self.get_optimizer(optimizer_name, learning_rate)
        loss_fn = self.get_loss_function(loss_name)

        # Set the model to training mode
        self.model.train()

        # Track the best train loss
        best_train_loss = float('inf')

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            train_loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            train_loss.backward()
            optimizer.step()

            # Update the best train loss
            if train_loss.item() < best_train_loss:
                best_train_loss = train_loss.item()

        # Return the best train loss for this run
        return best_train_loss

    
    def test(self, data, loss_name="cross_entropy"):
        # Get the loss function based on name
        loss_fn = self.get_loss_function(loss_name)

        # Set the model to evaluation mode
        self.model.eval()

        # Perform the forward pass for predictions
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            test_loss = loss_fn(out[data.test_mask], data.y[data.test_mask])

        return test_loss.item()  # Return the test loss




    def hyperparameter_search(self, data, layer_range, hidden_channels_range, epoch_range, activation_options, optimizer, loss, learning_rate, num_heads=None):
        """
        Perform hyperparameter search by iterating only over the parameters where min and max values are different.
        """
        # Determine whether the ranges are different
        vary_layers = layer_range['min'] != layer_range['max']
        vary_hidden_channels = hidden_channels_range['min'] != hidden_channels_range['max']
        vary_epochs = epoch_range['min'] != epoch_range['max']

        # Default values in case the range is fixed
        num_layers_fixed = layer_range['min']
        hidden_channels_fixed = hidden_channels_range['min']
        epochs_fixed = epoch_range['min']

        # Loop over activation functions
        for activation_str in activation_options:
            activation_fn = get_activation_function(activation_str)

            # Loop over num_layers only if the range is variable
            if vary_layers:
                num_layers_values = range(layer_range['min'], layer_range['max'] + 1, layer_range['step'])
            else:
                num_layers_values = [num_layers_fixed]

            # Loop over hidden_channels only if the range is variable
            if vary_hidden_channels:
                hidden_channels_values = range(hidden_channels_range['min'], hidden_channels_range['max'] + 1, hidden_channels_range['step'])
            else:
                hidden_channels_values = [hidden_channels_fixed]

            # Loop over epochs only if the range is variable
            if vary_epochs:
                epochs_values = range(epoch_range['min'], epoch_range['max'] + 1, epoch_range['step'])
            else:
                epochs_values = [epochs_fixed]

            # Perform the hyperparameter search
            for num_layers in num_layers_values:
                for hidden_channels in hidden_channels_values:
                    for epochs in epochs_values:
                        print(f"Training with {num_layers} layers, {hidden_channels} hidden channels, {epochs} epochs, "
                              f"{activation_str} activation, learning rate {learning_rate}")
                        
                        # Update model kwargs
                        self.hidden_channels = hidden_channels
                        self.model_kwargs['num_layers'] = num_layers
                        self.model_kwargs['activation'] = activation_fn

                        # Include num_heads if it's a GAT model
                        if self.model_name == 'GAT':
                            self.model_kwargs['num_heads'] = num_heads
                        
                        # Rebuild the model with updated hyperparameters
                        self.model = self.build_model()

                        # Train the model and get the best train loss
                        best_train_loss = self.train(data, optimizer_name=optimizer, loss_name=loss, epochs=epochs, learning_rate=learning_rate)

                        # Get the test loss after training
                        test_loss = self.test(data, loss_name=loss)

                        # Record the best train loss and the test loss for this configuration
                        if vary_hidden_channels:
                            self.plotter.record("hidden_channels", hidden_channels, best_train_loss, test_loss)
                        if vary_layers:
                            self.plotter.record("layers", num_layers, best_train_loss, test_loss)
                        if vary_epochs:
                            self.plotter.record("epochs", epochs, best_train_loss, test_loss)
                            
                        # Save the results to a CSV file
                        results = [
                            {"layers": num_layers, "hidden_channels": hidden_channels, "epochs": epochs,
                             "best_train_loss": best_train_loss, "test_loss": test_loss},
                        ]
                        save_training_results(self.output_path, results)

        # After the search is done, plot the results
        self.plotter.plot()


# Function to load hyperparameters from the config file
def load_hyperparameters(config_file):
    with open(config_file, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams

# Function to convert string activation to function
# Function to save training results to a CSV file

def save_training_results(filename, results):
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['layers', "hidden_channels", "epochs", 'best_train_loss', 'test_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        
        for result in results:
            writer.writerow(result)
            
def get_activation_function(activation_str):
    activations = {
        "relu": F.relu,
        "elu": F.elu,
        "leaky_relu": F.leaky_relu,
        # Add other activations if needed
    }
    return activations.get(activation_str, F.relu)  # Default to ReLU if not specified

def load_cora_dataset(batch_size=32):
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


# Example usage
if __name__ == "__main__":
    # Load hyperparameters from the config file
    hyperparams = load_hyperparameters('config.json')
    
    output_file = 'training_results.csv'

    # Assume we have a dataset (replace this with actual dataset)
    #edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    #x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float)
    #y = torch.tensor([0, 1, 0, 1])  # Dummy labels
    #train_mask = torch.tensor([True, True, False, False])
    #test_mask = torch.tensor([False, False, True, True])
    
    loader = load_cora_dataset(batch_size=32)
    
    edge_index = loader.dataset[0].edge_index
    x = loader.dataset[0].x
    y = loader.dataset[0].y
    train_mask = loader.dataset[0].train_mask
    test_mask = loader.dataset[0].test_mask
    

    ## print dimensions of the dataset
    #for data in loader:
        #print(data)
        #break
    
    data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    # Dynamically set the number of input features and output channels based on data
    num_features = data.num_node_features  # Set dynamically based on the data
    out_channels = len(set(data.y.numpy()))  # Number of unique labels

    # Get the model hyperparameters from the config file
    model_type = hyperparams["model_type"]
    
    # Instantiate the ModelWrapper with the loaded hyperparameters
    wrapper = ModelWrapper(
        model_name=model_type,
        num_features=num_features,
        out_channels=out_channels,
        hidden_channels=hyperparams['hidden_channels_range']['min'],# Start with the min hidden channels
        output_path = output_file, # Save the output to a file
        dropout=hyperparams.get("dropout", 0.5)  # Use dropout from config
    )
    
    # Perform hyperparameter search based on ranges from the config
    wrapper.hyperparameter_search(
        data=data,
        layer_range=hyperparams["layer_range"],
        hidden_channels_range=hyperparams["hidden_channels_range"],
        epoch_range=hyperparams["epoch_range"],
        activation_options=hyperparams["activation_options"],
        optimizer=hyperparams["optimizer"],
        loss=hyperparams["loss"],
        learning_rate=hyperparams["learning_rate"],
        num_heads=hyperparams.get("num_heads", 1)  # Use num_heads if it's a GAT model
    )