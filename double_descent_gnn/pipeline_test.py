import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch_geometric

# GCN model definition
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, num_layers=2, activation=F.relu, dropout=0.0):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        
        # Initialize the first convolutional layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))

        # Add intermediate convolutional layers if needed
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # Apply the GCN layers and the activation function
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        return x

# GAT model definition
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, num_heads=1, num_layers=2, activation=F.elu, dropout=0.0):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        
        self.gat_layers = torch.nn.ModuleList()
        self.gat_layers.append(GATConv(num_features, hidden_channels, heads=num_heads))
        for i in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads))
        self.gat_layers.append(GATConv(hidden_channels * num_heads, out_channels, heads=num_heads, concat=False))

    def forward(self, x, edge_index):
        for layer in self.gat_layers[:-1]:
            x = layer(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat_layers[-1](x, edge_index)
        return x

# Wrapper class to handle different model creation, training, and testing
class ModelWrapper:
    def __init__(self, model_name, num_features, out_channels, hidden_channels, **kwargs):
        # Dictionary mapping model names to their constructors
        self.model_constructors = {
            'GCN': GCN,
            'GAT': GAT,
        }
        
        # Ensure the model name is valid
        if model_name not in self.model_constructors:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(self.model_constructors.keys())}.")
        
        # Store model type
        self.model_name = model_name
        self.num_features = num_features  # Number of input features depends on data
        self.out_channels = out_channels  # Number of output channels depends on data
        self.hidden_channels = hidden_channels  # Set hidden_channels dynamically

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
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:  # Print every 10 epochs
                print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    def test(self, data, loss_name="cross_entropy"):
        # Get the loss function based on name
        loss_fn = self.get_loss_function(loss_name)

        # Set the model to evaluation mode
        self.model.eval()
        
        # Perform the forward pass for predictions
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            loss = loss_fn(out[data.test_mask], data.y[data.test_mask])
            pred = out.argmax(dim=1)  # Get predictions
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            accuracy = int(correct) / int(data.test_mask.sum())  # Accuracy calculation
        
        print(f'Test Loss: {loss.item()}')
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return accuracy

    # Method to loop through different hyperparameter combinations
    def hyperparameter_search(self, data, layer_range, hidden_channels_range, epoch_range, activation_options, optimizer, loss, learning_rate, num_heads=None):
        for num_layers in range(layer_range['min'], layer_range['max'] + 1, layer_range['step']):
            for hidden_channels in range(hidden_channels_range['min'], hidden_channels_range['max'] + 1, hidden_channels_range['step']):
                for epochs in range(epoch_range['min'], epoch_range['max'] + 1, epoch_range['step']):
                    for activation_str in activation_options:
                        activation_fn = get_activation_function(activation_str)
                        
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

                        # Train the model with the current hyperparameters
                        self.train(data, optimizer_name=optimizer, loss_name=loss, epochs=epochs, learning_rate=learning_rate)

                        # Test the model with the current hyperparameters
                        self.test(data, loss_name=loss)

# Function to load hyperparameters from the config file
def load_hyperparameters(config_file):
    with open(config_file, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams

# Function to convert string activation to function
def get_activation_function(activation_str):
    activations = {
        "relu": F.relu,
        "elu": F.elu,
        "leaky_relu": F.leaky_relu,
        # Add other activations if needed
    }
    return activations.get(activation_str, F.relu)  # Default to ReLU if not specified

# Example usage
if __name__ == "__main__":
    # Load hyperparameters from the config file
    hyperparams = load_hyperparameters('config.json')

    # Assume we have a dataset (replace this with actual dataset)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float)
    y = torch.tensor([0, 1, 0, 1])  # Dummy labels
    train_mask = torch.tensor([True, True, False, False])
    test_mask = torch.tensor([False, False, True, True])
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
        hidden_channels=hyperparams['hidden_channels_range']['min'],  # Start with the min hidden channels
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
