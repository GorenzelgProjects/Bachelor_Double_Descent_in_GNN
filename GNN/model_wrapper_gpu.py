import json
import torch
import torch.nn.functional as F
from models.conventional_models import GCN, GAT, GraphSAGE, GPRGNN, BLOCK_APPNP
from utils.custom_loader import OGBLoader
from plot import Plotter
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
#from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.utils import to_dense_adj
from madgap import MadGapRegularizer, MadValueCalculator
import csv
import os
import numpy as np
from oversmooth import OversmoothMeasure

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv

#from dataloader import add_label_noise, load_cora_dataset

from typing import Optional, Union

import torch
import torch.linalg as TLA
from torch import Tensor
import argparse

np.random.seed(42)
torch.manual_seed(42)

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Model Wrapper for GNN with Hyperparameter Search")
    parser.add_argument('--config', type=str, default='config_1.json', required=True, help='Path to the configuration file')
    return parser.parse_args()


# Wrapper class to handle different model creation, training, and testing
class ModelWrapper:
    '''
        This class is a wrapper around the PyTorch model classes. It allows for easy model creation, training, and testing.
        The class also includes a hyperparameter search method to loop through different hyperparameter combinations.

        Parameters:
            model_name (str): Name of the model to use. Supported models are 'GAT' and 'GCN'.
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

    def __init__(self, 
                 model_name, 
                 num_features, 
                 out_channels, 
                 hidden_channels, 
                 output_path, 
                 device, 
                 measure,
                 save_interval,
                 ppnp, 
                 K, 
                 alpha, 
                 dprate,
                 skip,
                 use_ppr, 
                 **kwargs):
        
        
        # Dictionary mapping model names to their constructors
        self.model_constructors = {
            'GCN': GCN,
            'GAT': GAT,
            'GRAPHSAGE': GraphSAGE,
            'GPRGNN': GPRGNN,
            'BLOCK_APPNP': BLOCK_APPNP
        }

        # Initialize Plotter
        self.plotter = Plotter()

        # Store model type
        self.model_name = model_name
        self.num_features = num_features
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.output_path = output_path
        self.device = device
        
        self.mse_loss = F.mse_loss
        
        #self.mad_gap_regularizer = mad_gap_regularizer
        #self.mad_value_calculator = mad_value_calculator
        self.measure = measure
        self.save_interval = save_interval
        self.ppnp = ppnp
        self.K = K
        self.alpha = alpha
        self.dprate = dprate
        self.skip = skip
        
        self.use_ppr = use_ppr
        
        if self.use_ppr:
            adj_matrix = csr_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0].numpy(), data.edge_index[1].numpy())), shape=(data.num_nodes, data.num_nodes))
            self.ppr_matrix = compute_personalized_pagerank(adj_matrix, alpha=self.alpha, top_k=self.K)
        else:
            self.ppr_matrix = None

        # Store the arguments needed for model initialization
        self.model_kwargs = kwargs

        # Build the model
        self.model = self.build_model().to(self.device)

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

    def calculate_accuracy(self, logits, labels):
        _, preds = torch.max(logits, dim=1)
        correct = (preds == labels).sum().item()
        accuracy = correct / len(labels)
        return accuracy
    
    def build_adj_mat(self, data):
        hidden_representations = self.get_hidden_representations(data)
        
        # Create adjacency matrix based on graph structure
        adjacency_matrix = torch.zeros((hidden_representations.shape[0], hidden_representations.shape[0]))
        adjacency_matrix[data.edge_index[0], data.edge_index[1]] = 1
        
        
        rmt_adjacency_matrix = torch.ones((hidden_representations.shape[0], hidden_representations.shape[0])) - adjacency_matrix
        
        adjacency_matrix = adjacency_matrix.cpu().numpy()
        rmt_adjacency_matrix = rmt_adjacency_matrix.cpu().numpy()
        
        return adjacency_matrix, rmt_adjacency_matrix
    
    def get_hidden_representations(self, data):
        self.model.eval()
        with torch.no_grad():
            #hidden_representations = model(data).detach().cpu().numpy()
            try:
                hidden_representations = self.model.forward_2(data.x.to(self.device), data.edge_index.to(self.device), self.ppr_matrix).detach().cpu().numpy()
                
            except:
                hidden_representations = self.model(data.x.to(self.device), data.edge_index.to(self.device), self.ppr_matrix).detach().cpu().numpy()
        
        return hidden_representations

    def train(self, data, optimizer_name="adam", loss_name="cross_entropy", epochs=100, learning_rate=0.001, num_layers=2, hidden_channels=16):
        
        # Get the optimizer and loss function based on names
        optimizer = self.get_optimizer(optimizer_name, learning_rate)
        loss_fn = self.get_loss_function(loss_name)
        
        # Also use a MSE loss
        #loss_MSE = F.mse_loss
        
        # ALso use a log_softmax loss
        loss_log_softmax = F.log_softmax
        
        # Also use a NLL loss
        loss_NLL = F.nll_loss
        
        # get model type:
        model_type = self.model_name

        # Set the model to training mode
        self.model.train()

        # Track the best train loss
        best_train_loss = float('inf')

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device), ppr_matrix=self.ppr_matrix)
            train_loss = loss_fn(out[data.train_mask.to(self.device)], data.y[data.train_mask].to(self.device))
            train_loss.backward()
            optimizer.step()
            
            # train loss as a float
            train_loss_cpu = train_loss.item()
            
            # print dimensions of out and data.y
            #print(out.shape)
            #print(data.y.shape)

                
            
            train_accuracy = self.calculate_accuracy(out[data.train_mask.to(self.device)], data.y[data.train_mask].to(self.device))
            
            # Calculate the MadValue regularization term
            #out_np = out.detach().cpu().numpy()  # Convert tensor to numpy array
            #mad_value = self.mad_value_calculator(out)
            
            
            # Calculate the MadGap regularization term
            #mad_gap_reg = self.mad_gap_regularizer(out)
            #print('MadGap:', mad_gap_reg.item())
            

            #print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item()}, Train Accuracy: {train_accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

            # Update the best train loss
            if train_loss.item() < best_train_loss:
                best_train_loss = train_loss.item()
                
                
            if (epoch+1) % self.save_interval == 0:
                # Get hidden representations
                hidden_representations = self.get_hidden_representations(data)
                #self.model.train()
                
                with torch.no_grad():
                # make the labels one-hot
                    out_mse = self.model.forward_2(data.x.to(self.device), data.edge_index.to(self.device), self.ppr_matrix)
                    # Apply softmax to get probabilities
                    out_mse = F.softmax(out_mse, dim=1)
                    one_hot_labels = F.one_hot(data.y[data.train_mask].to(self.device), num_classes=out_mse.shape[1])
                    train_mse_loss = self.mse_loss(out_mse[data.train_mask.to(self.device)], one_hot_labels.float())
                    train_mse_loss = train_mse_loss.item()
                    
                test_loss, test_mse_loss, test_accuracy = self.test(data, out_mse, loss_name=loss_name)
                
                # Calculate the MadValue
                mad_value = self.measure.get_mad_value(hidden_representations, self.adj_matrix, distance_metric='cosine', digt_num=4, target_idx=None)
                
                # Calculate the Mad_remote
                mad_rmt = self.measure.get_mad_value(hidden_representations, self.rmt_adj_matrix, distance_metric='cosine', digt_num=4, target_idx=None)
                
                # Calculate the MadGap
                madgap_value = round(float(mad_rmt - mad_value), 4)
                
                
                # Calculate the Dirichlet Energy
                dirichlet_energy = self.measure.get_dirichlet_energy(hidden_representations, self.adj_matrix)
                dirichlet_energy = round(float(dirichlet_energy), 4)
                
                # Save the results to a CSV file
                results = [
                    {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                        "train_loss": train_loss_cpu, "train_mse_loss": train_mse_loss, "train_accuracy": train_accuracy, 
                        "test_loss": test_loss, "test_mse_loss": test_mse_loss, "test_accuracy": test_accuracy, 
                        "mad_value": mad_value, "mad_gap": madgap_value, "dirichlet_energy": dirichlet_energy},
                ]
                save_training_results(self.output_path, results)
                
            
            
        #mad_value = mean_average_distance(out, edge_index=data.edge_index, adj_dict=self.adj_matrix, inverse=False)
        #print('MadValue_1:', mad_value)
        
        #mad_value = self.mad_value_calculator(out)
        #print('MadValue_2:', mad_value)
        

        # Return the best train loss for this run
        return best_train_loss

    
    def test(self, data, hidden_representations, loss_name="cross_entropy"):
        # Get the loss function based on name
        loss_fn = self.get_loss_function(loss_name)

        # Set the model to evaluation mode
        self.model.eval()

        # Perform the forward pass for predictions
        with torch.no_grad():
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device), ppr_matrix=self.ppr_matrix)
            test_loss = loss_fn(out[data.test_mask.to(self.device)], data.y[data.test_mask].to(self.device))
            test_accuracy = self.calculate_accuracy(out[data.test_mask.to(self.device)], data.y[data.test_mask].to(self.device))
            
            # make the labels one-hot
            one_hot_labels = F.one_hot(data.y[data.test_mask].to(self.device), num_classes=hidden_representations.shape[1])
            test_mse_loss = self.mse_loss(hidden_representations[data.test_mask.to(self.device)], one_hot_labels.float())

        return test_loss.item(), test_mse_loss.item(), test_accuracy


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
        
        # Get adjacency matrix & remote adjacency matrix
        self.adj_matrix, self.rmt_adj_matrix = self.build_adj_mat(data)

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
                hidden_channels_values = range(hidden_channels_range['min'], 
                                               hidden_channels_range['max'] + 1, 
                                               hidden_channels_range['step'])
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
                        self.model_kwargs['skip'] = self.skip

                        # Include num_heads if it's a GAT model
                        if self.model_name == 'GAT':
                            self.model_kwargs['num_heads'] = num_heads
                            
                        # Include extra for GPRGNN
                        elif self.model_name == 'GPRGNN' or self.model_name == 'BLOCK_APPNP':
                            self.model_kwargs['ppnp'] = self.ppnp
                            self.model_kwargs['K'] = self.K
                            self.model_kwargs['alpha'] = self.alpha
                            self.model_kwargs['dprate'] = self.dprate
                        
                        # Rebuild the model with updated hyperparameters
                        self.model = self.build_model().to(self.device)
                        
                        #print('Model:', self.model)

                        # Train the model and get the best train loss
                        best_train_loss = self.train(data, 
                                                     optimizer_name=optimizer, 
                                                     loss_name=loss, 
                                                     epochs=epochs, 
                                                     learning_rate=learning_rate, 
                                                     num_layers=num_layers, 
                                                     hidden_channels=hidden_channels)

                        # Get the test loss and accuracy after training
                        #test_loss, test_mse_loss, test_accuracy = self.test(data, loss_name=loss)

                        # Record the best train loss and the test loss for this configuration
                        #if vary_hidden_channels:
                            #self.plotter.record("hidden_channels", hidden_channels, best_train_loss, test_loss)
                        #if vary_layers:
                            #self.plotter.record("layers", num_layers, best_train_loss, test_loss)
                        #if vary_epochs:
                            #self.plotter.record("epochs", epochs, best_train_loss, test_loss)

        # After the search is done, plot the results
        #self.plotter.plot()


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
        fieldnames = ['model_type', 'layers', "hidden_channels", "epochs", 
                      'train_loss', 'train_mse_loss', 'train_accuracy', 
                      'test_loss', 'test_mse_loss', 'test_accuracy', 
                      'mad_value', 'mad_gap', 'dirichlet_energy']
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

def add_label_noise(labels, noise_level=0.1):
    num_classes = labels.max().item() + 1
    num_noise = int(noise_level * len(labels))
    noise_idx = torch.randperm(len(labels))[:num_noise]
    noise_labels = torch.randint(0, num_classes, (num_noise,))
    labels[noise_idx] = noise_labels
    return labels

def add_label_noise_multi_hot(labels, noise_level=0.1):
    """
    Add noise to multi-hot labels.
    :param labels: A multi-hot tensor of shape (num_nodes, num_classes).
    :param noise_level: Proportion of nodes to introduce noise to.
    :return: Noisy labels with flipped bits.
    """
    # Determine the number of nodes and labels to corrupt
    num_nodes, num_classes = labels.size()
    num_noise = int(noise_level * num_nodes)
    
    # Randomly select nodes to introduce noise
    noise_idx = torch.randperm(num_nodes)[:num_noise]
    
    # Flip random bits in the multi-hot labels
    noisy_labels = labels.clone()
    for idx in noise_idx:
        # Select a random class index to flip
        random_class = torch.randint(0, num_classes, (1,))
        noisy_labels[idx, random_class] = 1 - noisy_labels[idx, random_class]
    
    return noisy_labels

def compute_personalized_pagerank(adj_matrix, alpha=0.85, top_k=32):
    """
    Compute a sparse Personalized PageRank (PPR) matrix.
    Args:
        adj_matrix (scipy.sparse.csr_matrix): Adjacency matrix of the graph.
        alpha (float): Teleport probability.
        top_k (int): Number of top neighbors to retain per node.
    Returns:
        torch.Tensor: Sparse PPR matrix.
    """
    num_nodes = adj_matrix.shape[0]
    degree_matrix = csr_matrix(np.diag(np.array(adj_matrix.sum(axis=1)).flatten()))
    identity_matrix = csr_matrix(np.eye(num_nodes))
    ppr_matrix = alpha * inv(identity_matrix - (1 - alpha) * inv(degree_matrix).dot(adj_matrix))

    # Retain only top_k entries per row
    for i in range(num_nodes):
        row = ppr_matrix[i].toarray().flatten()
        top_indices = np.argsort(row)[-top_k:]
        mask = np.zeros_like(row, dtype=bool)
        mask[top_indices] = True
        row[~mask] = 0
        ppr_matrix[i] = csr_matrix(row)

    return torch.tensor(ppr_matrix.toarray(), dtype=torch.float)



# Example usage
if __name__ == "__main__":
    # Load hyperparameters from the config file
    checkpoint_dir = '/dtu/blackhole/10/141264/Bachelor_Double_Descent_in_GNN/GNN_double_descent/checkpoints'
    try:
        config_file = parse_args().config
        
    except SystemExit:
        #config_file = 'GNN_double_descent/config_1.json'
        config_file = 'GNN_double_descent/configs/config_test.json'
        
        print('Using default config file:', config_file)
    
    hyperparams = load_hyperparameters(config_file)
    
    # get the output path from the hyperparameters
    output_path = hyperparams["output_path"]
    output_file = os.path.join(checkpoint_dir, output_path)
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    osm = OversmoothMeasure()
    
    # Load the Cora dataset
    if hyperparams.get("dataset") == "cora":
        loader = load_cora_dataset(batch_size=32)
        
        edge_index = loader.dataset[0].edge_index
        x = loader.dataset[0].x
        y = loader.dataset[0].y
        
        y_train = y[loader.dataset[0].train_mask]
    
        # add label noise to the train labels
        y_train = add_label_noise(y_train, noise_level=hyperparams.get("label_noise", 0.15))
        
        y[loader.dataset[0].train_mask] = y_train
        
        train_mask = loader.dataset[0].train_mask
        test_mask = loader.dataset[0].test_mask
        
        
    elif hyperparams.get("dataset") == "ogbn-proteins":
        data_raw = torch.load("/dtu/blackhole/10/141264/Bachelor_Double_Descent_in_GNN/GNN_double_descent/datasets/ogbn-proteins.pt")
        dataset = OGBLoader(data_raw)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    #loader = load_cora_dataset(batch_size=32)
    
        edge_index = loader.dataset[0].edge_index
        x = loader.dataset[0].x
        y = loader.dataset[0].y
        
        y_train = y[loader.dataset[0].train_mask]
        
        # add label noise to the trains multi-hot labels
        y_train = add_label_noise_multi_hot(y_train, noise_level=hyperparams.get("label_noise", 0.15))
        y[loader.dataset[0].train_mask] = y_train
    
        train_mask = loader.dataset[0].train_mask
        test_mask = loader.dataset[0].test_mask
    
    # Create the data object
    data = torch_geometric.data.Data(x=x, 
                                     edge_index=edge_index, 
                                     y=y, 
                                     train_mask=train_mask, 
                                     test_mask=test_mask)

    # Dynamically set the number of input features and output channels based on data
    num_features = data.num_node_features  # Set dynamically based on the data
    
    
    # print the first row of x
    print(x[0])
    
    try:
        out_channels = len(set(data.y.numpy()))  # Number of unique labels
    except:
        out_channels = data.y.size(1)

    # Get the model hyperparameters from the config file
    model_type = hyperparams["model_type"]
    
    ppnp = hyperparams["ppnp"]
    K = hyperparams["K"]
    alpha = hyperparams["alpha"]
    dprate = hyperparams["dprate"]
    
    skip = True if hyperparams.get("skip")=="True" else False
    ppr = True if hyperparams.get("ppr")=="True" else False
    

    # Instantiate the ModelWrapper with the loaded hyperparameters
    # Instantiate the ModelWrapper with the loaded hyperparameters
    wrapper = ModelWrapper(
        model_name=model_type,
        num_features=num_features,
        out_channels=out_channels,
        hidden_channels=hyperparams['hidden_channels_range']['min'],# Start with the min hidden channels
        output_path=output_file, # Save the output to a file
        device=device,
        measure = osm,
        dropout=hyperparams.get("dropout", 0.5),  # Use dropout from config
        save_interval=hyperparams.get("save_interval", 10),  # Save interval from config
        ppnp=ppnp,
        K=K,
        alpha=alpha,
        dprate=dprate,
        skip=skip,  # Use skip from config
        use_ppr=ppr
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
