import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class MadGapRegularizer:
    def __init__(self, neb_mask, rmt_mask, target_idx, weight=0.01, device='cpu'):
        """
        Initializes the MadGapRegularizer with neighbor and remote masks, target indices, regularization weight, and device.

        Parameters:
        - neb_mask (torch.Tensor): Mask matrix for neighboring relations, [node_num * node_num]
        - rmt_mask (torch.Tensor): Mask matrix for remote relations, [node_num * node_num]
        - target_idx (torch.Tensor): Indices of the target nodes to calculate the mad_gap
        - weight (float): Weight applied to the MAD Gap regularization term
        - device (str): Device to run the computations on ('cpu' or 'cuda')
        """
        self.neb_mask = neb_mask.to(device)
        self.rmt_mask = rmt_mask.to(device)
        self.target_idx = target_idx.to(device)
        self.weight = weight
        self.device = device

    def __call__(self, intensor):
        """
        Computes the MAD Gap regularization value for the given input tensor.

        Parameters:
        - intensor (torch.Tensor): Node feature matrix, [node_num * hidden_dim]

        Returns:
        - mad_gap (torch.Tensor): The computed MAD Gap regularization value
        """
        intensor = intensor.to(self.device)
        node_num, feat_num = intensor.size()
        
        input1 = intensor.expand(node_num, node_num, feat_num)
        input2 = input1.transpose(0, 1)

        input1 = input1.contiguous().view(-1, feat_num)
        input2 = input2.contiguous().view(-1, feat_num)

        simi_tensor = F.cosine_similarity(input1, input2, dim=1, eps=1e-8).view(node_num, node_num)
        dist_tensor = 1 - simi_tensor

        neb_dist = torch.mul(dist_tensor, self.neb_mask)
        rmt_dist = torch.mul(dist_tensor, self.rmt_mask)
        
        divide_neb = (neb_dist != 0).sum(1).type(torch.FloatTensor).to(self.device) + 1e-8
        divide_rmt = (rmt_dist != 0).sum(1).type(torch.FloatTensor).to(self.device) + 1e-8

        neb_mean_list = neb_dist.sum(1) / divide_neb
        rmt_mean_list = rmt_dist.sum(1) / divide_rmt

        neb_mad = torch.mean(neb_mean_list[self.target_idx])
        rmt_mad = torch.mean(rmt_mean_list[self.target_idx])

        mad_gap = rmt_mad - neb_mad
        return self.weight * mad_gap
    
    
    
class MadValueCalculator:
    def __init__(self, mask_arr, distance_metric='cosine', digt_num=4, target_idx=None):
        """
        Initializes the MadValueCalculator with mask array, distance metric, digit number, and target indices.

        Parameters:
        - mask_arr (np.ndarray): Mask matrix for relations, [node_num * node_num]
        - distance_metric (str): Distance metric to use for pairwise distances
        - digt_num (int): Number of digits to round the MAD value
        - target_idx (np.ndarray or None): Indices of the target nodes to calculate the mad value
        """
        self.mask_arr = mask_arr
        self.distance_metric = distance_metric
        self.digt_num = digt_num
        self.target_idx = target_idx

    def __call__(self, in_arr):
        """
        Computes the MAD value for the given input array.

        Parameters:
        - in_arr (np.ndarray): Node feature matrix, [node_num * hidden_dim]

        Returns:
        - mad (float): The computed MAD value
        """
        
        # Normalize the input array
        #in_arr = in_arr / (np.linalg.norm(in_arr, axis=1, keepdims=True) + 1e-8)
        
        #H_norm = F.normalize(in_arr, p=2, dim=1)
        #cosine_similarity = torch.mm(H_norm, H_norm.t())
        #dist_arr = 1 - cosine_similarity  # Cosine distance matrix
        
        # Convert to numpy array
        #dist_arr = dist_arr.cpu().detach().numpy()
        
        #dist_arr = pairwise_distances(in_arr, in_arr, metric=self.distance_metric)
        
        #mask_dist = np.multiply(dist_arr, self.mask_arr)

        #divide_arr = (mask_dist != 0).sum(1) + 1e-8
        #node_dist = mask_dist.sum(1) / divide_arr

        #if self.target_idx is None:
            #mad = np.mean(node_dist)
        #else:
            #node_dist = np.multiply(node_dist, self.target_idx)
            #mad = node_dist.sum() / ((node_dist != 0).sum() + 1e-8)

        #mad = round(mad, self.digt_num)
        #return mad
        
        """
        Computes the Mean Average Distance (MAD) for a given graph representation H and target mask M_tgt.

        Args:
            H (torch.Tensor): Node representations of shape (n, d), where n is the number of nodes and d is the embedding dimension.
            M_tgt (torch.Tensor): Target mask of shape (n, n), with 1s indicating target node pairs and 0s elsewhere.

        Returns:
            float: The Mean Average Distance (MAD) value.
        """
        # Step 1: Compute the cosine distance matrix D
        # Normalize H to compute cosine similarity and then derive cosine distance
        
    
        H = in_arr
        M_tgt = self.mask_arr
        
        #H_norm = F.normalize(H, p=2, dim=1)
        #H_norm = H
        #cosine_similarity = torch.mm(H_norm, H_norm.t())
        #D = 1 - cosine_similarity  # Cosine distance matrix
        
        
        # Calculate the norms for each row
        norms = torch.norm(H, dim=1)
        
        # Calculate the dot product matrix between rows of H
        dot_product_matrix = H @ H.T
        
        # Outer product of norms to get the denominator matrix for cosine similarity
        denominator_matrix = norms.unsqueeze(1) * norms.unsqueeze(0)
        
        # Calculate cosine similarity matrix
        cosine_similarity_matrix = dot_product_matrix / denominator_matrix
        
        # Convert cosine similarity to cosine distance
        D = 1 - cosine_similarity_matrix

        # Step 2: Apply target mask M_tgt to get D_tgt
        D_tgt = D * M_tgt
    
    
        # Step 3: Compute row-wise average distance for non-zero values
        non_zero_counts = torch.sum(D_tgt > 0, dim=1)  # Count of non-zero elements per row
        row_sums = torch.sum(D_tgt, dim=1)  # Sum of distances per row
        avg_distances = row_sums / non_zero_counts  # Row-wise average distance
        
        # Handle any division by zero (if there are rows with no non-zero entries)
        avg_distances[non_zero_counts == 0] = 0  # Set to 0 where there are no valid target pairs
        
        # Step 4: Calculate MAD_tgt
        valid_row_count = torch.sum(non_zero_counts > 0)  # Count of rows with non-zero values
        MAD_tgt = torch.sum(avg_distances) / valid_row_count  # Mean of average distances

        return MAD_tgt.item()


        # Step 3: Compute per-node average distance for target pairs
        # Use masking to avoid division by zero
        #non_zero_mask = (D_tgt > 0).float()
        #per_node_avg_dist = torch.sum(D_tgt, dim=1) / torch.sum(non_zero_mask, dim=1)

        # Step 4: Compute MAD by averaging per-node distances for target pairs
        #MAD_tgt = torch.mean(per_node_avg_dist[torch.isfinite(per_node_avg_dist)])  # Avoid NaNs

        #return MAD_tgt.item()
        