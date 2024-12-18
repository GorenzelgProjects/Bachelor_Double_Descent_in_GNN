import json
import torch
import torch.nn.functional as F
from models.conventional_models import GCN, GAT, GPRGNN
from plot import Plotter
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from madgap import MadGapRegularizer, MadValueCalculator
import csv
import os
import numpy as np
#from dataloader import add_label_noise, load_cora_dataset

from typing import Optional, Union

import torch
import torch.linalg as TLA
from torch import Tensor
import argparse
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
                 mad_gap_regularizer,
                 mad_value_calculator,
                 adj_dict,
                 save_interval,
                 ppnp, 
                 K, 
                 alpha, 
                 dprate, 
                 **kwargs):
        # Dictionary mapping model names to their constructors
        self.model_constructors = {
            'GCN': GCN,
            'GAT': GAT,
            'GPRGNN': GPRGNN,
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
        self.mad_gap_regularizer = mad_gap_regularizer
        self.mad_value_calculator = mad_value_calculator
        self.adj_dict = adj_dict
        self.save_interval = save_interval
        self.ppnp = ppnp
        self.K = K
        self.alpha = alpha
        self.dprate = dprate

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

    def train(self, data, optimizer_name="adam", loss_name="cross_entropy", epochs=100, learning_rate=0.001, num_layers=2, hidden_channels=16):
        # Get the optimizer and loss function based on names
        optimizer = self.get_optimizer(optimizer_name, learning_rate)
        loss_fn = self.get_loss_function(loss_name)
        
        # Also use a MSE loss
        loss_MSE = F.mse_loss
        
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
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            train_loss = loss_fn(out[data.train_mask.to(self.device)], data.y[data.train_mask].to(self.device))
            train_loss.backward()
            optimizer.step()
            
            # train loss as a float
            train_loss_cpu = train_loss.item()
            
            # print dimensions of out and data.y
            #print(out.shape)
            #print(data.y.shape)
            
            train_accuracy = self.calculate_accuracy(out[data.train_mask.to(self.device)], data.y[data.train_mask].to(self.device))
            test_loss, test_accuracy = self.test(data, loss_name=loss_name)
            
            # Calculate the MadValue regularization term
            out_np = out.detach().cpu().numpy()  # Convert tensor to numpy array
            #mad_value = self.mad_value_calculator(out)
            
            
            # Calculate the MadGap regularization term
            #mad_gap_reg = self.mad_gap_regularizer(out)
            #print('MadGap:', mad_gap_reg.item())
            

            #print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item()}, Train Accuracy: {train_accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

            # Update the best train loss
            if train_loss.item() < best_train_loss:
                best_train_loss = train_loss.item()
                
                
            if (epoch+1) % self.save_interval == 0:
                # Calculate the MadValue regularization term

                mad_value = mean_average_distance(out, edge_index=data.edge_index, adj_dict=self.adj_dict, inverse=False)
                # Save the results to a CSV file
                results = [
                    {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                        "train_loss": train_loss_cpu, "train_accuracy": train_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy, "mad_value": mad_value},
                ]
                save_training_results(self.output_path, results)
            
        #mad_value = mean_average_distance(out, edge_index=data.edge_index, adj_dict=self.adj_matrix, inverse=False)
        #print('MadValue_1:', mad_value)
        
        #mad_value = self.mad_value_calculator(out)
        #print('MadValue_2:', mad_value)
        

        # Return the best train loss for this run
        return best_train_loss

    
    def test(self, data, loss_name="cross_entropy"):
        # Get the loss function based on name
        loss_fn = self.get_loss_function(loss_name)

        # Set the model to evaluation mode
        self.model.eval()

        # Perform the forward pass for predictions
        with torch.no_grad():
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            test_loss = loss_fn(out[data.test_mask.to(self.device)], data.y[data.test_mask].to(self.device))
            test_accuracy = self.calculate_accuracy(out[data.test_mask.to(self.device)], data.y[data.test_mask].to(self.device))

        return test_loss.item(), test_accuracy


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

                        # Include num_heads if it's a GAT model
                        if self.model_name == 'GAT':
                            self.model_kwargs['num_heads'] = num_heads
                            
                        # Include extra for GPRGNN
                        elif self.model_name == 'GPRGNN':
                            self.model_kwargs['ppnp'] = self.ppnp
                            self.model_kwargs['K'] = self.K
                            self.model_kwargs['alpha'] = self.alpha
                            self.model_kwargs['dprate'] = self.dprate
                        
                        # Rebuild the model with updated hyperparameters
                        self.model = self.build_model().to(self.device)

                        # Train the model and get the best train loss
                        best_train_loss = self.train(data, 
                                                     optimizer_name=optimizer, 
                                                     loss_name=loss, 
                                                     epochs=epochs, 
                                                     learning_rate=learning_rate, 
                                                     num_layers=num_layers, 
                                                     hidden_channels=hidden_channels)

                        # Get the test loss and accuracy after training
                        test_loss, test_accuracy = self.test(data, loss_name=loss)

                        # Record the best train loss and the test loss for this configuration
                        #if vary_hidden_channels:
                            #self.plotter.record("hidden_channels", hidden_channels, best_train_loss, test_loss)
                        #if vary_layers:
                            #self.plotter.record("layers", num_layers, best_train_loss, test_loss)
                        #if vary_epochs:
                            #self.plotter.record("epochs", epochs, best_train_loss, test_loss)

        # After the search is done, plot the results
        #self.plotter.plot()



def build_adj_dict(num_nodes: int, edge_index: Tensor, inverse: bool) -> dict:
    r"""A function to turn a list of edges (edge_index) into an adjacency list,
    stored in a dictionary with vertex numbers as keys and lists of adjacent
    nodes as values.

    Args:
        num_nodes (int): number of nodes
        edge_index (torch.Tensor): edge list

    :rtype: dict
    """
    # initialize adjacency dict with empty neighborhoods for all nodes
    adj_dict: dict = {nodeid: [] for nodeid in range(num_nodes)}
    
    
    if not inverse:
        for eidx in range(edge_index.shape[1]):
            ctail, chead = edge_index[0, eidx].item(), edge_index[1, eidx].item()

            if chead not in adj_dict[ctail]:
                adj_dict[ctail].append(chead)
    
    else:
        for eidx in range(edge_index.shape[1]):
            ctail, chead = edge_index[0, eidx].item(), edge_index[1, eidx].item()
            
            remote_nodes = [node for node in range(num_nodes) if node != ctail]
            # remove all values from chead
            remote_nodes = [node for node in remote_nodes if node != chead]
            
            adj_dict[ctail].append(remote_nodes)
            
            

    return adj_dict


@torch.no_grad()
def dirichlet_energy(
    feat_matrix: Tensor,
    edge_index: Optional[Tensor] = None,
    adj_dict: Optional[dict] = None,
    p: Optional[Union[int, float]] = 2,
    inverse: bool = False,
) -> float:
    r"""The 'Dirichlet Energy' node similarity measure from the
    `"A Survey on Oversmoothing in Graph Neural Networks"
    <https://arxiv.org/abs/2303.10993>`_ paper.

    .. math::
        \mu\left(\mathbf{X}^n\right) =
        \sqrt{\mathcal{E}\left(\mathbf{X}^n\right)}

    with

    .. math::
        \mathcal{E}(\mathbf{X}^n) = \mathrm{Ave}_{i\in \mathcal{V}}
        \sum_{j \in \mathcal{N}_i} ||\mathbf{X}_{i}^n - \mathbf{X}_{j}^n||_p ^2

    Args:
        feat_matrix (torch.Tensor): The node feature matrix.
        edge_index (torch.Tensor, optional): The edge list
            (default: :obj:`None`)
        adj_dict (dict, optional): The adjacency dictionary
            (default: :obj:`None`)
        p (int or float, optional): The order of the norm (default: :obj:`2`)

    :rtype: float
    """
    num_nodes: int = feat_matrix.shape[0]
    de: Tensor = torch.tensor(0, dtype=torch.float32)

    if adj_dict is None:
        if edge_index is None:
            raise ValueError("Neither 'edge_index' nor 'adj_dict' was provided.")
        adj_dict = build_adj_dict(num_nodes=num_nodes, edge_index=edge_index, inverse=inverse)

    def inner(x_i: Tensor, x_js: Tensor) -> Tensor:
        return TLA.vector_norm(x_i - x_js, ord=p, dim=1).square().sum()

    for node_index in range(num_nodes):
        own_feat_vector = feat_matrix[[node_index], :]
        nbh_feat_matrix = feat_matrix[adj_dict[node_index], :]

        de += inner(own_feat_vector, nbh_feat_matrix).cpu()

    return torch.sqrt(de / num_nodes).item()


@torch.no_grad()
def mean_average_distance(
    feat_matrix: Tensor,
    edge_index: Optional[Tensor] = None,
    adj_dict: Optional[dict] = None,
    inverse: bool = False,
) -> float:
    r"""The 'Mean Average Distance' node similarity measure from the
    `"A Survey on Oversmoothing in Graph Neural Networks"
    <https://arxiv.org/abs/2303.10993>`_ paper.

    .. math::
        \mu(\mathbf{X}^n) = \mathrm{Ave}_{i\in \mathcal{V}}
        \sum_{j \in \mathcal{N}_i}
        \frac{{\mathbf{X}_i ^n}^\mathrm{T}\mathbf{X}_j ^n}
        {||\mathbf{X}_i ^n|| ||\mathbf{X}_j^n||}

    Args:
        feat_matrix (torch.Tensor): The node feature matrix.
        edge_index (torch.Tensor, optional): The edge list
            (default: :obj:`None`)
        adj_dict (dict, optional): The adjacency dictionary
            (default: :obj:`None`)

    :rtype: float
    """
    num_nodes: int = feat_matrix.shape[0]
    mad: Tensor = torch.tensor(0, dtype=torch.float32)

    if adj_dict is None:
        if edge_index is None:
            raise ValueError("Neither 'edge_index' nor 'adj_dict' was provided.")
        adj_dict = build_adj_dict(num_nodes=num_nodes, edge_index=edge_index, inverse=inverse)
        
    # print the first 10 values of adj_dict
    #print(list(adj_dict.items())[:10])

    def inner(x_i: Tensor, x_js: Tensor) -> Tensor:
        return (
            1
            - (x_i @ x_js.t())
            / (TLA.vector_norm(x_i, ord=2) * TLA.vector_norm(x_js, ord=2, dim=1))
        ).sum()

    for node_index in range(num_nodes):
        own_feat_vector = feat_matrix[[node_index], :]
        nbh_feat_matrix = feat_matrix[adj_dict[node_index], :]

        mad += inner(own_feat_vector, nbh_feat_matrix).cpu()

    return (mad / num_nodes).item()

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
        fieldnames = ['model_type', 'layers', "hidden_channels", "epochs", 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy', 'mad_value']
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


# Example usage
if __name__ == "__main__":
    # Load hyperparameters from the config file
    checkpoint_dir = '/dtu/blackhole/10/141264/Bachelor_Double_Descent_in_GNN/GNN_double_descent/checkpoints'
    try:
        config_file = parse_args().config
        
    except SystemExit:
        print("Error: Please provide the path to the configuration file using --config argument.")
        exit(1)
    
    hyperparams = load_hyperparameters(config_file)
    
    # get the output path from the hyperparameters
    output_path = hyperparams["output_path"]
    output_file = os.path.join(checkpoint_dir, output_path)
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Assume we have a dataset (replace this with actual dataset)
    #edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    #x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float)
    #y = torch.tensor([0, 1, 0, 1])  # Dummy labels
    #train_mask = torch.tensor([True, True, False, False])
    #test_mask = torch.tensor([False, False, True, True])
    
    # Load the Cora dataset
    loader = load_cora_dataset(batch_size=32)
    
    edge_index = loader.dataset[0].edge_index
    x = loader.dataset[0].x
    y = loader.dataset[0].y
    
    y = add_label_noise(y, noise_level=0.1)
    
    train_mask = loader.dataset[0].train_mask
    test_mask = loader.dataset[0].test_mask
    
    #print(edge_index.size())
    #print(train_mask.size())
    
    # Generate neb_mask and rmt_mask for neighboring and remote relations
    node_num = x.size(0)
    
    #print(node_num)
    
    adj_dict = build_adj_dict(num_nodes=node_num, edge_index=edge_index, inverse=False)
    
    #print(list(adj_dict.items())[2])
    
    #adj_dict_inverted = build_adj_dict(num_nodes=node_num, edge_index=edge_index, inverse=True)
    
    #print(list(adj_dict_inverted.items())[2])
    
    
    neb_mask = torch.eye(node_num)  # Assume self-loops as neighbors for simplicity
    #mask_arr = np.eye(node_num)  # Identity matrix for simplicity
    rmt_mask = 1 - neb_mask  # Remote nodes as those not directly connected
    
    
    adj_matrix = to_dense_adj(edge_index)
    
    # remove first dimension (1, num_nodes, num_nodes) to  (num_nodes, num_nodes)
    adj_matrix = adj_matrix.squeeze(0)
    
    num_nodes = adj_matrix.size(0)
    
    #print(adj_matrix.size())
    
    # Initialize M_tgt as the adjacency matrix
    # Option 1: If you want MAD for neighbors, use the adjacency matrix as-is
    neighbor_mask = adj_matrix.clone().float()
    
    # Option 2: If you want MAD for remote nodes, use the complement of the adjacency matrix
    #remote_mask = (1 - adj_matrix).float()

    # Set the diagonal to zero for both masks (a node is not a neighbor or remote to itself)
    neighbor_mask.fill_diagonal_(0)
    #remote_mask.fill_diagonal_(0)
    
    # Convert the neighbor mask to a tensor
    neighbor_mask = torch.tensor(neighbor_mask, dtype=torch.float).to(device)
    
    
    # Create the mask array which should be the same size as the node_num * node_num and contains 1s for neighboring nodes and 0s for remote nodes which is given in the edge_index
    mask_arr = np.zeros((node_num, node_num))
    for i in range(edge_index.size(1)):
        mask_arr[edge_index[0][i]][edge_index[1][i]] = 1
        mask_arr[edge_index[1][i]][edge_index[0][i]] = 1
        
    # Convert the mask array to a tensor
    #mask_arr = torch.tensor(mask_arr, dtype=torch.float).to(device)
    

    # Target indices (all nodes for this example)
    target_idx = torch.arange(node_num)
    
    # Create the data object
    data = torch_geometric.data.Data(x=x, 
                                     edge_index=edge_index, 
                                     y=y, 
                                     train_mask=train_mask, 
                                     test_mask=test_mask)

    # Dynamically set the number of input features and output channels based on data
    num_features = data.num_node_features  # Set dynamically based on the data
    out_channels = len(set(data.y.numpy()))  # Number of unique labels

    # Get the model hyperparameters from the config file
    model_type = hyperparams["model_type"]
    
    ppnp = hyperparams["ppnp"]
    K = hyperparams["K"]
    alpha = hyperparams["alpha"]
    dprate = hyperparams["dprate"]
    
    
    # Instantiate the MadValueCalculator
    mad_value_calculator = MadValueCalculator(mask_arr=neighbor_mask, 
                                              distance_metric='cosine', 
                                              digt_num=4, 
                                              target_idx=target_idx.numpy())

    
    # Create the MadGapRegularizer object
    mad_gap_regularizer = MadGapRegularizer(neb_mask=neb_mask, 
                                            rmt_mask=rmt_mask, 
                                            target_idx=target_idx, 
                                            weight=0.01,
                                            device=device)

    # Instantiate the ModelWrapper with the loaded hyperparameters
    # Instantiate the ModelWrapper with the loaded hyperparameters
    wrapper = ModelWrapper(
        model_name=model_type,
        num_features=num_features,
        out_channels=out_channels,
        hidden_channels=hyperparams['hidden_channels_range']['min'],# Start with the min hidden channels
        output_path=output_file, # Save the output to a file
        device=device,
        mad_gap_regularizer=mad_gap_regularizer,  # Pass the MadGapRegularizer object
        mad_value_calculator=mad_value_calculator,  # Pass the MadValueCalculator object
        dropout=hyperparams.get("dropout", 0.5),  # Use dropout from config
        adj_dict=adj_dict,
        save_interval=hyperparams.get("save_interval", 10),  # Save interval from config
        ppnp=ppnp,
        K=K,
        alpha=alpha,
        dprate=dprate
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
