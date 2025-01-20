import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_add_pool
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv


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
    
    def __init__(self, num_features, hidden_channels, out_channels, num_layers=2, activation=F.relu, dropout=0.0, skip=False, use_ppr=False):
        super(GCN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.skip = skip
        self.use_ppr = use_ppr
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

    def __init__(self, num_features, hidden_channels, out_channels, num_heads=1, num_layers=2, activation=F.elu, dropout=0.0, skip=False, use_ppr=False):
        super(GAT, self).__init__()
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

    def __init__(self, num_features, hidden_channels, out_channels, num_layers=2, activation=F.relu, dropout=0.0, skip=False, use_ppr=False):
        super(GraphSAGE, self).__init__()
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
    def __init__(self, num_features, hidden_channels, out_channels, num_layers=2, activation=F.relu, dropout=0.0, ppnp='PPNP', K=10, alpha=0.1, dprate=0.0, skip=False, use_ppr=False):
        super(GPRGNN, self).__init__()
        
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
    def __init__(self, num_features, hidden_channels, out_channels, num_layers=2, activation=F.relu, dropout=0.0, ppnp='PPNP', K=10, alpha=0.1, dprate=0.0, skip=False, use_ppr=False):
        super(BLOCK_APPNP, self).__init__()
        
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
    
class GCNGPP(nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, num_layers=3, dropout=0.5, skip=True, use_ppr=True):
        super(GCNGPP, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.skip = skip
        self.use_ppr = use_ppr

        # Define GCN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))  # Input layer
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))  # Hidden layers
        self.convs.append(GCNConv(hidden_channels, hidden_channels))  # Output GCN layer

        # Preprocessing layer for skip connections
        if skip:
            self.preprocess = create_ffn(num_features, hidden_channels, dropout)

        # PPR feature projection
        if use_ppr:
            self.ppr_weight = torch.nn.Parameter(torch.Tensor(num_features, hidden_channels))
            torch.nn.init.xavier_uniform_(self.ppr_weight)
            

        # Final output layer
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, ppr_matrix=None):
        if self.skip:
            x_skip = self.preprocess(x)

        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:
                x = x + x_skip

        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features

        # Pooling for graph-level representation
        x = global_add_pool(x, batch)

        # Final linear layer
        x = self.linear(x)
        return x
    
    def forward_2(self, x, edge_index, batch, ppr_matrix=None):
        if self.skip:
            x_skip = self.preprocess(x)

        if self.use_ppr and ppr_matrix is not None:
            ppr_features = ppr_matrix @ x
            ppr_features = ppr_features @ self.ppr_weight

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:
                x = x + x_skip

        if self.use_ppr and ppr_matrix is not None:
            x = x + ppr_features

        # Pooling for graph-level representation
        x = global_add_pool(x, batch)

        # Final linear layer
        x = self.linear(x)
        return x
    
    
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

