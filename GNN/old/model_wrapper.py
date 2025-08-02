from models.conventional_models import GCN, GAT, GraphSAGE, GPRGNN, BLOCK_APPNP, GCNGPP, GSGPP, GATGPP

import torch
import torch.nn.functional as F

import numpy as np

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv

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
                 data, 
                 model_name, 
                 **kwargs):
        
        
        # Dictionary mapping model names to their constructors
        self.model_constructors = {
            'GCN': GCN,
            'GAT': GAT,
            'GRAPHSAGE': GraphSAGE,
            'GPRGNN': GPRGNN,
            'BLOCK_APPNP': BLOCK_APPNP,
            'GCNGPP': GCNGPP,
            'GSGPP': GSGPP,
            'GATGPP': GATGPP
        }

        # Store the arguments needed for model initialization
        self.model_kwargs = kwargs
        
        
        self.model_name = model_name
        self.num_layers = kwargs['num_layers']
        self.num_features = kwargs['num_features']
        self.out_channels = kwargs['out_channels']
        self.hidden_channels = kwargs['hidden_channels']
        self.dropout = kwargs['dropout']
        self.device = kwargs['device']
        activation_name = kwargs.get("activation_name", "relu")
        loss_name = kwargs.get("loss_name", "cross_entropy")
        optimizer_name = kwargs.get("optimizer_name", "adam")
        learning_rate = kwargs.get("learning_rate", 0.01)
        
        activation_name = str(activation_name[0])
        
        self.loss = self.get_loss_function(loss_name)
        self.activation = self.get_activation_function(activation_name)
        
        self.mse_loss = F.mse_loss
        # Build the model
        self.model = self.build_model().to(self.device)
        
        self.optimizer = self.get_optimizer(optimizer_name, learning_rate)

        #self.ppnp = kwargs.get("ppnp", False)
        #self.K = kwargs.get("K", 32)
        #self.alpha = kwargs.get("alpha", 0.1)
        #self.dprate = kwargs.get("dprate", 0.5)
        #self.skip = kwargs.get("skip", False)
        
        self.use_ppr = kwargs.get("use_ppr", False)
        
        if self.use_ppr:
            adj_matrix = csr_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0].numpy(), data.edge_index[1].numpy())), shape=(data.num_nodes, data.num_nodes))
            self.ppr_matrix = self.compute_personalized_pagerank(adj_matrix, alpha=self.alpha, top_k=self.K)
        else:
            self.ppr_matrix = None
            
        

        # Store the arguments needed for model initialization
        self.model_kwargs = kwargs

        
        
        
    def build_model(self):
        # Get the model constructor based on the model name
        model_constructor = self.model_constructors[self.model_name]
        
        # Create an instance of the model by passing the keyword arguments
        model = model_constructor(**self.model_kwargs)
        
        return model
    
    def get_optimizer(self, optimizer_name, learning_rate):
        if optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
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
        
    def get_activation_function(self, activation_name):
        activations = {
            "relu": F.relu,
            "elu": F.elu,
            "leaky_relu": F.leaky_relu,
            # Add other activations if needed
        }
        try:
            return activations.get(activation_name, F.relu)
        
        except KeyError:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        
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
        adj_matrix += 1e-10 * sparse.eye(adj_matrix.shape[0])
        
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