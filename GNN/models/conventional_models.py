import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, SAGEConv, APPNP
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, JumpingKnowledge
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import degree
import math


class GCN(torch.nn.Module):
    """
        Graph Convolutional Network (GCN) model.

        Parameters:
            num_features (int): Number of input features.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output channels.
            num_layers (int, optional): Number of GCN layers. Default is 2.
            activation (callable, optional): Activation function. Default is F.relu.
            dropout (float, optional): Dropout rate. Default is 0.0.

        Returns:
            torch.Tensor: Output tensor after passing through the GCN layers.
    """
    
    def __init__(self, **kwargs):
        super(GCN, self).__init__() 
        
        num_features = kwargs.get("num_features", 1)
        out_channels = kwargs.get("out_channels", 8)
        num_layers = kwargs.get("num_layers", 2)
        self.activation = kwargs.get("activation", F.relu)
        self.dropout = kwargs.get("dropout", 0.0)
        self.skip = kwargs.get("skip", False)
        self.use_ppr = kwargs.get("use_ppr", False)
        hidden_channels = kwargs.get("hidden_channels", 64)
        # Initialize the first convolutional layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))

        # Add intermediate convolutional layers if needed
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        if self.skip:
            # Create a process layer.
            self.preprocess = create_ffn(num_features, hidden_channels, self.dropout)
            
        if self.use_ppr:
            self.ppr_weight = torch.nn.Parameter(torch.Tensor(num_features, hidden_channels))
            torch.nn.init.xavier_uniform_(self.ppr_weight)
    

    def forward(self, x, edge_index, ppr_matrix=None):
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight    
            
        # Apply the GCN layers and the activation function
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:
                # print 10 samples of x
                #print(x[:10])
                #print(x_skip[:10])
                
                x = x + x_skip
        
        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features
        
        return F.log_softmax(x, dim=1)
    
    def forward_2(self, x, edge_index, ppr_matrix=None):
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight
        # Apply the GCN layers and the activation function
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:
                x = x + x_skip
        
        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features
            
        return x


class GAT(torch.nn.Module):
    """
        Graph Attention Network (GAT) model.

        Parameters:
            num_features (int): Number of input features.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output channels.
            num_heads (int, optional): Number of attention heads. Default is 1.
            num_layers (int, optional): Number of GAT layers. Default is 2.
            activation (callable, optional): Activation function. Default is F.elu.
            dropout (float, optional): Dropout rate. Default is 0.0.

        Returns:
            torch.Tensor: Output tensor after passing through the GAT layers.
    """

    def __init__(self, **kwargs):
        super(GAT, self).__init__()
        
        num_features = kwargs.get("num_features", 1)
        out_channels = kwargs.get("out_channels", 8)
        num_layers = kwargs.get("num_layers", 2)
        activation = kwargs.get("activation", F.elu)
        dropout = kwargs.get("dropout", 0.0)
        skip = kwargs.get("skip", False)
        use_ppr = kwargs.get("use_ppr", False)
        hidden_channels = kwargs.get("hidden_channels", 64)
        num_heads = kwargs.get("num_heads", 1)
        
        self.num_features = num_features
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.skip = skip
        self.use_ppr = use_ppr
        
        self.gat_layers = torch.nn.ModuleList()
        self.gat_layers.append(GATConv(num_features, hidden_channels, heads=num_heads))
        for i in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads))
        self.gat_layers.append(GATConv(hidden_channels * num_heads, out_channels, heads=num_heads, concat=False))
        
        if self.skip:
            # Create a process layer.
            self.preprocess = create_ffn(num_features, hidden_channels * num_heads, dropout)
            
        #if self.use_ppr:
            #self.ppr_weight = torch.nn.Parameter(torch.Tensor(num_features, hidden_channels))
            #torch.nn.init.xavier_uniform_(self.ppr_weight)
            
    def forward(self, x, edge_index, ppr_matrix=None):
        
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            #ppr_features = ppr_features @ self.ppr_weight
        
        for layer in self.gat_layers[:-1]:
            x = layer(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:
                x = x + x_skip
        x = self.gat_layers[-1](x, edge_index)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features
        
        return F.log_softmax(x, dim=1)
    
    def forward_2(self, x, edge_index, ppr_matrix=None):
        
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            #ppr_features = ppr_features @ self.ppr_weight
        
        for layer in self.gat_layers[:-1]:
            x = layer(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:
                x = x + x_skip
        x = self.gat_layers[-1](x, edge_index)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features
        
        return x
    
class GraphSAGE(torch.nn.Module):
    """
        GraphSAGE model.

        Parameters:
            num_features (int): Number of input features.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output channels.
            num_layers (int, optional): Number of GraphSAGE layers. Default is 2.
            activation (callable, optional): Activation function. Default is F.relu.
            dropout (float, optional): Dropout rate. Default is 0.0.

        Returns:
            torch.Tensor: Output tensor after passing through the GraphSAGE layers.
    """

    def __init__(self, **kwargs):
        super(GraphSAGE, self).__init__()
        
        num_features = kwargs.get("num_features", 1)
        out_channels = kwargs.get("out_channels", 8)
        num_layers = kwargs.get("num_layers", 2)
        activation = kwargs.get("activation", F.relu)
        dropout = kwargs.get("dropout", 0.0)
        skip = kwargs.get("skip", False)
        use_ppr = kwargs.get("use_ppr", False)
        hidden_channels = kwargs.get("hidden_channels", 64)
        
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.skip = skip
        self.use_ppr = use_ppr
        
        # Initialize the first GraphSAGE layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_channels))

        # Add intermediate GraphSAGE layers if needed
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        if self.skip:
            # Create a process layer for skip connections
            self.preprocess = create_ffn(num_features, hidden_channels, dropout)
            
        if self.use_ppr:
            self.ppr_weight = torch.nn.Parameter(torch.Tensor(num_features, hidden_channels))
            torch.nn.init.xavier_uniform_(self.ppr_weight)

    def forward(self, x, edge_index, ppr_matrix=None):
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight
        # Apply the GraphSAGE layers and the activation function
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:
                x = x + x_skip

        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features

        return F.log_softmax(x, dim=1)
    
    def forward_2(self, x, edge_index, ppr_matrix=None):
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight
        # Apply the GraphSAGE layers and the activation function
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:
                x = x + x_skip

        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features

        return x
    

    
class GPRGNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GPRGNN, self).__init__()
        
        num_features = kwargs.get("num_features", 1)
        out_channels = kwargs.get("out_channels", 8)
        num_layers = kwargs.get("num_layers", 2)
        activation = kwargs.get("activation", F.relu)
        dropout = kwargs.get("dropout", 0.0)
        skip = kwargs.get("skip", False)
        use_ppr = kwargs.get("use_ppr", False)
        hidden_channels = kwargs.get("hidden_channels", 64)
        ppnp = kwargs.get("ppnp", 'PPNP')
        K = kwargs.get("K", 10)
        alpha = kwargs.get("alpha", 0.1)
        dprate = dropout
        
        
        self.skip = skip
        self.use_ppr = use_ppr
        self.gprgnn_layers = torch.nn.ModuleList()
        
        self.gprgnn_layers.append(Linear(num_features, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.gprgnn_layers.append(Linear(hidden_channels, hidden_channels))
        
        self.gprgnn_layers.append(Linear(hidden_channels, out_channels))

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        #elif ppnp == 'GPR_prop':
            #self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        #self.Init = Init
        self.dprate = dprate
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        
        for layer in self.gprgnn_layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.gprgnn_layers[-1](x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        
class BLOCK_APPNP(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BLOCK_APPNP, self).__init__()
        
        num_features = kwargs.get("num_features", 1)
        out_channels = kwargs.get("out_channels", 8)
        num_layers = kwargs.get("num_layers", 2)
        activation = kwargs.get("activation", F.relu)
        dropout = kwargs.get("dropout", 0.0)
        skip = kwargs.get("skip", False)
        use_ppr = kwargs.get("use_ppr", False)
        hidden_channels = kwargs.get("hidden_channels", 64)
        ppnp = kwargs.get("ppnp", 'PPNP')
        K = kwargs.get("K", 10)
        alpha = kwargs.get("alpha", 0.1)
        dprate = dropout
        
        
        self.skip = skip
        self.gprgnn_layers = torch.nn.ModuleList()
        
        if num_layers == 1:
            self.gprgnn_layers.append(Linear(num_features, out_channels))
            self.gprgnn_layers.append(APPNP(K, alpha))
        
        else:
            self.gprgnn_layers.append(Linear(num_features, hidden_channels))
            self.gprgnn_layers.append(APPNP(K, alpha))
        
            for _ in range(num_layers - 2):
                self.gprgnn_layers.append(Linear(hidden_channels, hidden_channels))
                self.gprgnn_layers.append(APPNP(K, alpha))
        
            self.gprgnn_layers.append(Linear(hidden_channels, out_channels))
            self.gprgnn_layers.append(APPNP(K, alpha))
        
        
        #self.Init = Init
        self.dprate = dprate
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        
        for layer_index in range(1, len(self.gprgnn_layers), 2):
            x = self.gprgnn_layers[layer_index-1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            #if self.dprate == 0.0:
                #x = F.dropout(x, p=self.dprate, training=self.training)
                
            x = self.gprgnn_layers[layer_index](x, edge_index)
            
        return F.log_softmax(x, dim=1)
    
    
class GCNLayer(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNLayer, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(4, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
'''
class GCNGPP(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, num_layers=2, activation=F.relu, dropout=0.0, skip=False, use_ppr=False):
        super(GCNGPP, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.skip = skip
        self.use_ppr = use_ppr
        # Initialize the first convolutional layer
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(num_features, hidden_channels))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        
        # Add intermediate convolutional layers if needed
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        if self.skip:
            # Create a process layer.
            self.preprocess = create_ffn(num_features, hidden_channels, dropout)
            
        if self.use_ppr:
            self.ppr_weight = torch.nn.Parameter(torch.Tensor(num_features, hidden_channels))
            torch.nn.init.xavier_uniform_(self.ppr_weight)
        # Define GCN layers
        
        # Define a fully connected layer for graph classification
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        

    def forward(self, data, ppr_matrix=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight    
            
        # Apply the GCN layers and the activation function
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.batch_norms[i](x)
            if self.skip:                
                x = x + x_skip
            
        
        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[i](x)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features
            
        
        # Global mean pooling to aggregate node embeddings
        x = global_mean_pool(x, batch)
        #x = global_max_pool(x, batch)

        # Fully connected layer for graph classification
        x = self.fc(x)
        return x
    
    def forward_2(self, data, ppr_matrix=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight    
            
        # Apply the GCN layers and the activation function
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.batch_norms[i](x)
            if self.skip:                
                x = x + x_skip
        
        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[i](x)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features

        # Global mean pooling to aggregate node embeddings
        x = global_mean_pool(x, batch)

        # Fully connected layer for graph classification
        x = self.fc(x)
        return x
'''
class GSGPP(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GSGPP, self).__init__()
        
        num_features = kwargs.get("num_features", 1)
        out_channels = kwargs.get("out_channels", 8)
        num_layers = kwargs.get("num_layers", 2)
        hidden_channels = kwargs.get("hidden_channels", 64)
        activation = kwargs.get("activation", F.relu)
        dropout = kwargs.get("dropout", 0.0)
        skip = kwargs.get("skip", False)
        use_ppr = kwargs.get("use_ppr", False)
        
        self.num_features = num_features
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.skip = skip
        self.use_ppr = use_ppr
        # Initialize the first convolutional layer
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_channels))

        # Add intermediate convolutional layers if needed
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        if self.skip:
            # Create a process layer.
            self.preprocess = create_ffn(num_features, hidden_channels, dropout)
            
        if self.use_ppr:
            self.ppr_weight = torch.nn.Parameter(torch.Tensor(num_features, hidden_channels))
            torch.nn.init.xavier_uniform_(self.ppr_weight)
        # Define GCN layers
        
        # Define a fully connected layer for graph classification
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        

    def forward(self, data, ppr_matrix=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight    
            
        # Apply the GCN layers and the activation function
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:                
                x = x + x_skip
        
        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features

        # Global mean pooling to aggregate node embeddings
        x = global_mean_pool(x, batch)

        # Fully connected layer for graph classification
        x = self.fc(x)
        return x
    
    def forward_2(self, data, ppr_matrix=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight    
            
        # Apply the GCN layers and the activation function
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:                
                x = x + x_skip
        
        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features

        # Global mean pooling to aggregate node embeddings
        x = global_mean_pool(x, batch)

        # Fully connected layer for graph classification
        x = self.fc(x)
        return x
    
class GATGPP(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GATGPP, self).__init__()
        
        num_features = kwargs.get("num_features", 1)
        out_channels = kwargs.get("out_channels", 8)
        num_layers = kwargs.get("num_layers", 2)
        activation = kwargs.get("activation", F.elu)
        dropout = kwargs.get("dropout", 0.0)
        skip = kwargs.get("skip", False)
        use_ppr = kwargs.get("use_ppr", False)
        hidden_channels = kwargs.get("hidden_channels", 64)
        
        self.num_features = num_features
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.skip = skip
        self.use_ppr = use_ppr
        num_heads = 4
        # Initialize the first convolutional layer
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_channels, heads=num_heads))

        # Add intermediate convolutional layers if needed
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads))

        # Output layer
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=False))
        
        if self.skip:
            # Create a process layer.
            self.preprocess = create_ffn(num_features, hidden_channels, dropout)
            
        if self.use_ppr:
            self.ppr_weight = torch.nn.Parameter(torch.Tensor(num_features, hidden_channels))
            torch.nn.init.xavier_uniform_(self.ppr_weight)
        # Define GCN layers
        
        # Define a fully connected layer for graph classification
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        

    def forward(self, data, ppr_matrix=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight    
            
        # Apply the GCN layers and the activation function
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:                
                x = x + x_skip
        
        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features

        # Global mean pooling to aggregate node embeddings
        x = global_mean_pool(x, batch)

        # Fully connected layer for graph classification
        x = self.fc(x)
        return x
    
    def forward_2(self, data, ppr_matrix=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        if self.skip:
            x_skip = self.preprocess(x)
            
        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight    
            
        # Apply the GCN layers and the activation function
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:                
                x = x + x_skip
        
        # Output layer (no activation applied here)
        x = self.convs[-1](x, edge_index)
        
        # Combine with PPR features
        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features

        # Global mean pooling to aggregate node embeddings
        x = global_mean_pool(x, batch)

        # Fully connected layer for graph classification
        x = self.fc(x)
        return x
    


class GCNGPP(torch.nn.Module):

    def __init__(self, **kwargs):
        super(GCNGPP, self).__init__()
        
        num_features = kwargs.get("num_features", 1)
        out_channels = kwargs.get("out_channels", 8)
        num_layers = kwargs.get("num_layers", 2)
        hidden_channels = kwargs.get("hidden_channels", 64)
        activation = kwargs.get("activation", F.relu)
        dropout = kwargs.get("dropout", 0.0)
        skip = kwargs.get("skip", False)
        use_ppr = kwargs.get("use_ppr", False)
        
        
        num_layer = num_layers
        emb_dim = hidden_channels
        num_class = out_channels
        drop_ratio = dropout
        
        residual = False
        JK = "last"
        graph_pooling = "mean"
        
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling
        self.activation = activation

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        #if virtual_node:
            #self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        #else:
            #self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        #if graph_pooling == "set2set":
            #self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        #else:
        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)
            
            
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        
        self.skip = skip
        self.use_ppr = use_ppr

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(num_features, emb_dim) # uniform input node embedding

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):

            self.convs.append(GCNLayer(emb_dim))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            
        # Preprocessing layer for skip connections
        if skip:
            self.preprocess = create_ffn(emb_dim, emb_dim, drop_ratio)

        # PPR feature projection
        if use_ppr:
            self.ppr_weight = torch.nn.Parameter(torch.Tensor(emb_dim, emb_dim))
            torch.nn.init.xavier_uniform_(self.ppr_weight)
            
            

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # Convert torch x to long
        #x = x.long()
        
        # squeeze the batch dimension
        #x = x.squeeze()
        print(x.shape)
        print(edge_attr.shape)
        print(edge_index.shape)
        
        
        ### computing input node embedding

        h_list = [self.node_encoder(x)]
        
        if self.skip:
            x_skip = self.preprocess(self.node_encoder(x))
        
        for layer in range(self.num_layer):
            #print(h_list[layer].shape)
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(self.activation(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]
            
            if self.skip:
                h += x_skip

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        #return node_representation

        h_graph = self.pool(node_representation, batch)

        return self.graph_pred_linear(h_graph)


    
# FFN for skip connections
def create_ffn(num_features, hidden_channels, dropout_rate):
        layers = []
        ffn = torch.nn.Linear(num_features, hidden_channels)
        bn = torch.nn.BatchNorm1d(hidden_channels)

        layers.append(ffn)
        layers.append(bn)
        layers.append(torch.nn.ReLU())
        if dropout_rate > 0:
            layers.append(torch.nn.Dropout(dropout_rate))

        return torch.nn.Sequential(*layers)

# 

