import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys

sys.path.append("models/")
from mlp import MLP


class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim,input_dim_hyper, hidden_dim, output_dim, final_dropout, learn_eps,
                 graph_pooling_type, neighbor_pooling_type, device, multi_head=False):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()
        self.hyper_mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        self.hyper_batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                self.hyper_mlps.append(MLP(num_mlp_layers, input_dim_hyper, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, 2*hidden_dim, hidden_dim, hidden_dim))
                self.hyper_mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.hyper_batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(2*hidden_dim, output_dim))

        # self.att_a=nn.Parameter(torch.rand(1))

        if multi_head:
            self.atts = nn.Parameter(torch.Tensor(np.random.rand(self.num_layers, 2)))
        else:
            self.atts=nn.Parameter(torch.Tensor(np.random.rand(2,)))

        self.multi_head=multi_head


    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        # compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                # add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                # padding, dummy data is assumed to be stored in -1
                pad.extend([-1] * (max_deg - len(pad)))

                # Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device)

    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1. / len(graph.g)] * len(graph.g))

            else:
                ###sum pooling
                elem.extend([1] * len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes altogether

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def hyper_next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.hyper_mlps[layer](pooled)
        h = self.hyper_batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def hyper_next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes altogether

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # representation of neighboring and center nodes
        pooled_rep = self.hyper_mlps[layer](pooled)

        h = self.hyper_batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    # def attention_layer(self,h1,h2):
    #     a=nn.Parameter(torch.randn(1))
    #     b=nn.Parameter(torch.randn(1))
    #     sum=a+b
    #     return h1*a/sum+h2*a/sum

    def forward(self, batch_graph, batch_hyper_graph, batch_motif2A):
        X1_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)
        hyper_graph_pool= self.__preprocess_graphpool(batch_hyper_graph)
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block1 = self.__preprocess_neighbors_sumavepool(batch_graph)
        # list of hidden representation at each layer (including input)
        hidden_rep1 = [X1_concat]
        h1 = X1_concat

        X2_concat = torch.cat([graph.node_features for graph in batch_hyper_graph], 0).to(self.device)
        motif2A = np.zeros((X1_concat.shape[0], X2_concat.shape[0]))
        # print('mitif2A:',motif2A.shape)
        start_idx=0
        end_idx=0
        for graph_idx in range(len(batch_motif2A)):
            temp_motif2A=np.array(batch_motif2A[graph_idx])
            end_idx=start_idx+len(temp_motif2A)
            # print('temp_motif2A:{} start:{} end:{}'.format(temp_motif2A.shape,start_idx,end_idx))
            motif2A[start_idx:end_idx,start_idx:end_idx]=temp_motif2A
            start_idx=end_idx
        motif2A = torch.FloatTensor(motif2A.T).to(self.device)
        Adj_block_hyper = self.__preprocess_neighbors_sumavepool(batch_hyper_graph)
        h2 = X2_concat.to(torch.float32)

        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h1 = self.next_layer_eps(h1, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h1 = self.next_layer_eps(h1, layer, Adj_block=Adj_block1)
                if len(batch_hyper_graph) != 0:
                    h2 = self.hyper_next_layer_eps(h2, layer, Adj_block=Adj_block_hyper)
                # h2 = self.next_layer_eps(h2, layer, Adj_block=Adj_block2)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h1 = self.next_layer(h1, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h1 = self.next_layer(h1, layer, Adj_block=Adj_block1)
                if len(batch_hyper_graph) != 0:
                    h2 = self.hyper_next_layer(h2, layer, Adj_block=Adj_block_hyper)
                # h2 = self.next_layer(h2, layer, Adj_block=Adj_block2)

            if len(batch_hyper_graph) != 0:
                softmax = torch.nn.Softmax(0)
                if self.multi_head:
                    ab = softmax(self.atts[layer])
                else:
                    ab=softmax(self.atts)
                
                h22 = torch.mm(motif2A, h2)
                
                # h1=F.normalize(h1,dim=1)
                # h22=F.normalize(h22,dim=1)
                
                h22 = ab[1] * h22
                h1 = ab[0] * h1
                h1 = torch.cat((h1,h22),1)
            
            hidden_rep1.append(h1)
            #hidden_rep2.append(h2)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for layer, h1 in enumerate(hidden_rep1):
            pooled_h = torch.spmm(graph_pool, h1)
            #pooled_h2 = torch.spmm(hyper_graph_pool, hidden_rep2[layer])
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout,
                                          training=self.training)
            #score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h2), self.final_dropout,
                                          #training=self.training)

        return score_over_layer