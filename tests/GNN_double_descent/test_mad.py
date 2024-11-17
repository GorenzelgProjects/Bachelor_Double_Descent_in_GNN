from typing import Optional, Union
import json
import torch
import torch.nn.functional as F
from models.conventional_models import GCN, GAT
from plot import Plotter
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from madgap import MadGapRegularizer, MadValueCalculator
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

from typing import Optional, Union

import torch.linalg as TLA
from torch import Tensor

import torch
import torch.linalg as TLA
from torch import Tensor
import node_similarity as ns
import model_wrapper_gpu as mw
import madgap as mg

loader = mw.load_cora_dataset(batch_size=32)

edge_index = loader.dataset[0].edge_index
x = loader.dataset[0].x
y = loader.dataset[0].y
train_mask = loader.dataset[0].train_mask
test_mask = loader.dataset[0].test_mask
node_num = x.size(0)
adj_dict = ns.build_adj_dict(num_nodes=node_num, edge_index=edge_index)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


adj_matrix = to_dense_adj(edge_index)
    
adj_matrix = adj_matrix.squeeze(0)

num_nodes = adj_matrix.size(0)

neighbor_mask = adj_matrix.clone().float()
remote_mask = (1 - adj_matrix).float()
neighbor_mask.fill_diagonal_(0)
neighbor_mask.fill_diagonal_(0)


neighbor_mask = torch.tensor(neighbor_mask, dtype=torch.float).to(device)
remote_mask = torch.tensor(remote_mask, dtype=torch.float).to(device)
target_idx = torch.arange(node_num)
x = x.to(device)    

mad_value_calculator = mg.MadValueCalculator(mask_arr=neighbor_mask, 
                                              distance_metric='cosine', 
                                              digt_num=4, 
                                              target_idx=target_idx.numpy())


    
    # Create the MadGapRegularizer object
#mad_gap_regularizer = mg.MadGapRegularizer(neb_mask=neighbor_mask, 
                                      #  rmt_mask=remote_mask, 
                                     #   target_idx=target_idx, 
                                      #  weight=1,
                                      #  device=device)

#mad_2 = ns.mean_average_distance(feat_matrix=x, adj_dict=adj_dict)


"""
def mad_3(target_idx, mask_arr, in_arr, distance_metric='cosine'):
    # Normalize the input array
    in_arr = in_arr.cpu().detach().numpy() / (np.linalg.norm(in_arr.cpu().detach().numpy(), axis=1, keepdims=True) + 1e-8)
    
    H_norm = F.normalize(in_arr, p=2, dim=1)
    cosine_similarity = torch.mm(H_norm, H_norm.t())
    dist_arr = 1 - cosine_similarity  # Cosine distance matrix
    
     #Convert to numpy array
    dist_arr = dist_arr.cpu().detach().numpy()
    
    dist_arr = pairwise_distances(in_arr, in_arr, metric=distance_metric)
    
    mask_dist = np.multiply(dist_arr, mask_arr)

    divide_arr = (mask_dist != 0).sum(1) + 1e-8
    node_dist = mask_dist.sum(1) / divide_arr

    if target_idx is None:
        mad = np.mean(node_dist)
    else:
        node_dist = np.multiply(node_dist, target_idx)
        mad = node_dist.sum() / ((node_dist != 0).sum() + 1e-8)

    mad = round(mad, 4)
    return mad
"""
#mad_33 = mad_3(target_idx=target_idx, mask_arr=neighbor_mask, in_arr=x, distance_metric='cosine')

#Normalize the feature matrix
x = F.normalize(x, p=2, dim=1)

mad_2 = ns.mean_average_distance(feat_matrix=x, edge_index=edge_index)

#make a nice print to compare all the 3 values
print(f"Mad1: {mad_value_calculator(x)}")
print(f"Mad2: {mad_2}")
#print(f"Mad3: {mad_33}")
