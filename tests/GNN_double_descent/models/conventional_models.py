import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


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