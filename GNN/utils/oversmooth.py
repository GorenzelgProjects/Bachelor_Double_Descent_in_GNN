import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import pairwise_distances
#import seaborn as sns
import matplotlib.pyplot as plt

# Create a class called Oversmooth
class OversmoothMeasure:
    # Initialize the class
    def __init__(self):
        pass
            
    '''    
    the numpy version for mad (Be able to compute quickly)
    in_arr:[node_num * hidden_dim], the node feature matrix;
    mask_arr: [node_num * node_num], the mask matrix of the target raltion;
    target_idx = [1,2,3...n], the nodes idx for which we calculate the mad value;
    '''
    # MAD calculation
    def get_mad_value(self, in_arr, mask_arr, distance_metric='cosine', digt_num=4, target_idx=None):
        dist_arr = pairwise_distances(in_arr, in_arr, metric=distance_metric)
        
        # filter the target node pairs by element-wise multiplication with a mask-matrix
        mask_dist = np.multiply(dist_arr, mask_arr)
        
        #mask_dist = np.multiply(dist_arr, mask_arr)

        divide_arr = (mask_dist != 0).sum(1) + 1e-8

        node_dist = mask_dist.sum(1) / divide_arr
        
        if target_idx is None or not target_idx.any():
            mad = np.mean(node_dist)
        else:
            node_dist = np.multiply(node_dist, target_idx)
            mad = node_dist.sum() / ((node_dist != 0).sum() + 1e-8)

        mad = round(mad, digt_num)

        return mad
    
    '''
    the numpy version for the energy calculation
    input:
    hidden_representations: [node_num * hidden_dim], the hidden representation matrix;
    adjacency_matrix: [node_num * node_num], the adjacency matrix of the graph;
    
    output:
    energy: the energy value of the hidden representation matrix;
    '''
    def get_dirichlet_energy(self, hidden_representations, adjacency_matrix):
        # Compute Dirichlet Energy
        energy = 0.0
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] > 0:
                    diff = hidden_representations[i] - hidden_representations[j]
                    energy += np.dot(diff, diff) * adjacency_matrix[i, j]
        energy /= 2.0  # The energy is divided by 2 as it's symmetric
        return energy