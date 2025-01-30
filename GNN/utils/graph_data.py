import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Actor, Airports
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

import random
import numpy as np


class GraphDataset:
    def __init__(self, dataname='cora', raw_data_path=None, batch_size=32, shuffle=True, noise_level=0.0, test_size=0.2, val_size=0.1, random_state=42):
        """
        Initialize the GraphDataset class.
        :param data: Optional initial dataset (e.g., a dictionary of adjacency lists)
        """
        self.dataname = dataname
        self.raw_data_path = raw_data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise_level = noise_level
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def get_dataset(self):
        """
        Load a dataset into the GraphDataset instance.
        :param dataset: A dictionary representing the graph adjacency list
        """
        if self.dataname == 'cora':
            collected_data = self.get_cora()
        elif self.dataname == 'airports':
            collected_data = self.get_airports()
        elif self.dataname == 'citeseer':
            collected_data = self.get_citeseer()
            
        elif self.dataname == 'ogbn_proteins':
            if self.raw_data_path is None:
                collected_data = self.get_ogbn_proteins()
            else:
                raise ValueError("Raw data path must be provided for ogbn_proteins dataset.")
            
        elif self.dataname == 'ogbg_ppa':
            if self.raw_data_path is None:
                collected_data = self.get_ogbg_ppa()
            else:
                raise ValueError("Raw data path must be provided for ogbg_ppa dataset.")
            
        elif self.dataname == 'mutag':
            collected_data = self.get_mutag()
            
        else:
            raise ValueError(f"Dataset {self.dataname} not supported.")
        
        return collected_data
        
        
    def load_dataset(self, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return loader
    
        
    def add_label_noise(self, labels):
        num_classes = labels.max().item() + 1
        num_noise = int(self.noise_level * len(labels))
        noise_idx = torch.randperm(len(labels))[:num_noise]
        noise_labels = torch.randint(0, num_classes, (num_noise,))
        labels[noise_idx] = noise_labels
        return labels

    # Add label noise
    def add_label_noise_mutag(self, dataset):
        num_samples = len(dataset)
        num_noisy_samples = int(self.noise_level * num_samples)
        # Randomly select indices to flip labels
        noisy_indices = random.sample(range(num_samples), num_noisy_samples)
        
        # Flip the labels at the selected indices
        for idx in noisy_indices:
            dataset[idx].y = 1 - dataset[idx].y  # Flip label (assuming binary labels: 0 or 1)
            
        return dataset
    
    def add_zeros(data):
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
        return data
        
    def get_cora(self):
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        loader = self.load_dataset(dataset=dataset)
        
        edge_index = loader.dataset[0].edge_index
        x = loader.dataset[0].x
        y = loader.dataset[0].y
        
        y_train = y[loader.dataset[0].train_mask]
    
        # add label noise to the train labels
        if self.noise_level > 0:
            y_train = self.add_label_noise(y_train)
        
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
        
        evaluator = None
        
        try:
            out_channels = len(set(data.y.numpy()))  # Number of unique labels
        except:
            out_channels = data.y.size(1)
        
        collected_data = (data, num_features, out_channels, evaluator)
        
        return collected_data
    

    def get_citeseer(self):
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
        loader = self.load_dataset(dataset=dataset)
        
        edge_index = loader.dataset[0].edge_index
        x = loader.dataset[0].x
        y = loader.dataset[0].y
        
        y_train = y[loader.dataset[0].train_mask]
        
        if self.noise_level > 0:
            # add label noise to the train labels
            y_train = self.add_label_noise(y_train)
        
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
        
        evaluator = None
        # print the first row of x
        #print(x[0])
        
        try:
            out_channels = len(set(data.y.numpy()))  # Number of unique labels
        except:
            out_channels = data.y.size(1)
            
        collected_data = (data, num_features, out_channels, evaluator)
        
        return collected_data
    
    
    def get_airports(self):
        dataset = Airports(root='path/to/dataset', name='usa')
        data = dataset[0]  # Single graph
        
        # Step 2: Create Train/Validation/Test Masks
        num_nodes = data.num_nodes

        # Split indices into train, validation, and test
        train_idx, test_idx = train_test_split(range(num_nodes), test_size=self.test_size, random_state=self.random_state)
        train_idx, val_idx = train_test_split(train_idx, test_size=self.val_size, random_state=self.random_state)  # 20% test, 15% validation


        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # Add masks to the data object
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        loader = self.load_dataset(dataset=data)
        
        edge_index = loader.dataset.edge_index
        
        
        x = loader.dataset.x
        y = loader.dataset.y
        
        
        y_train = y[loader.dataset.train_mask]
    
        if self.noise_level > 0:
            # add label noise to the train labels
            y_train = self.add_label_noise(y_train)
        
        y[loader.dataset.train_mask] = y_train
        
        train_mask = loader.dataset.train_mask
        test_mask = loader.dataset.test_mask
        
        # Create the data object
        data = torch_geometric.data.Data(x=x, 
                                        edge_index=edge_index, 
                                        y=y, 
                                        train_mask=train_mask, 
                                        test_mask=test_mask)

        # Dynamically set the number of input features and output channels based on data
        num_features = data.num_node_features  # Set dynamically based on the data
        
        evaluator = None
        
        try:
            out_channels = len(set(data.y.numpy()))  # Number of unique labels
        except:
            out_channels = data.y.size(1)
            
        collected_data = (data, num_features, out_channels, evaluator)
        
        return collected_data
    
    def get_mutag(self):
        # Load MUTAG dataset
        dataset = TUDataset(root="data/TUDataset", name="MUTAG")

        # Split the dataset into train, validation, and test sets
        train_idx, test_idx = train_test_split(range(len(dataset)), test_size=self.test_size, random_state=42)
        train_idx, valid_idx = train_test_split(train_idx, test_size=self.val_size, random_state=42)

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        
        #print(test_dataset.y)
        
        if self.noise_level > 0:
            train_dataset = self.add_label_noise_mutag(train_dataset)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        data = [train_loader, valid_loader, test_loader]
        
        # Dynamically set the number of input features and output channels based on data
        num_features = dataset.num_node_features  # Set dynamically based on the data
        out_channels = dataset.num_classes  # Number of unique labels
        
        evaluator = None
        
        collected_data = (data, num_features, out_channels, evaluator)
        
        return collected_data
    
    def get_ogbn_proteins(self):
        data_raw = torch.load(self.raw_data_path)
        dataset = OGBLoader(data_raw)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        #loader = load_cora_dataset(batch_size=32)
    
        edge_index = loader.dataset[0].edge_index
        x = loader.dataset[0].x
        y = loader.dataset[0].y
        
        y_train = y[loader.dataset[0].train_mask]
        
        # add label noise to the trains multi-hot labels
        #y_train = add_label_noise_multi_hot(y_train, noise_level=hyperparams.get("label_noise", 0.15))
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
        
        evaluator = None
        
        try:
            out_channels = len(set(data.y.numpy()))  # Number of unique labels
        except:
            out_channels = data.y.size(1)
            
        collected_data = (data, num_features, out_channels, evaluator)
        
        return collected_data
    
    def get_ogbg_ppa(self):
        dataset_path = self.raw_data_path
        dataset = PygGraphPropPredDataset(name = "ogbg-ppa", root=dataset_path, transform = self.add_zeros)
        #dataset = PygGraphPropPredDataset(name = "ogbg-ppa", transform = add_zeros)

        split_idx = dataset.get_idx_split()

        ### automatic evaluator. takes dataset name as input
        evaluator = Evaluator("ogbg-ppa")

        #train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=0)
        #valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=0)
        #test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=0)
        
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=self.shuffle, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        data = [train_loader, valid_loader, test_loader]

        # Dynamically set the number of input features and output channels based on data
        num_features = 1  # Set dynamically based on the data
        out_channels = dataset.num_classes  # Number of unique labels
        
        collected_data = (data, num_features, out_channels, evaluator)
        
        return collected_data
    
    

    def __repr__(self):
        return f"GraphDataset({self.data})"



# Create a custom dataset class if needed
class OGBLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 1  # Single graph dataset

    def __getitem__(self, idx):
        return self.data