from model_wrapper import ModelWrapper
from utils.oversmooth import OversmoothMeasure

import numpy as np
import torch
import torch.nn.functional as F
import os
import csv

class Trainer:
    def __init__(self, 
                    data,
                    model_name,
                    num_features, 
                    out_channels, 
                    device,
                    output_path,
                    hyperparams,
                    evaluator,
                    gpp,
                    skip,
                    ppr,
                    **kwargs):
        
        
        self.data = data
        self.model_name = model_name
        self.num_features = num_features
        self.out_channels = out_channels
        self.device = device
        self.output_path = output_path
        self.evaluator=evaluator
        
        self.gpp = gpp
        self.skip = skip
        self.ppr = ppr
        
        self.layer_range = hyperparams.get("layer_range", {"min": 2, "max": 4, "step": 1})
        self.hidden_channels_range = hyperparams.get("hidden_channels_range", {"min": 1, "max": 64, "step": 1})
        self.epoch_range = hyperparams.get("epoch_range", {"min": 1000, "max": 1000, "step": 100})
                                           
        self.activation_name = hyperparams.get("activation_options", "relu")
        self.optimizer_name = hyperparams.get("optimizer", "adam")
        self.loss_name = hyperparams.get("loss", "cross_entropy")
        self.learning_rate = hyperparams.get("learning_rate", 0.01)
        
        self.num_heads = hyperparams.get("num_heads", 8)
        self.ppnp = hyperparams.get("ppnp", False)
        self.K = hyperparams.get("K", 10)
        self.alpha = hyperparams.get("alpha", 0.1)
        self.dprate = hyperparams.get("dropout", 0.5)
        self.save_interval = hyperparams.get("save_interval", 10)

        # Store the arguments needed for model initialization
        self.model_kwargs = kwargs
        
        self.measure = OversmoothMeasure()    
        
        
    def hyperparameter_train(self):
        """
        Perform hyperparameter search by iterating only over the parameters where min and max values are different.
        """
        # Determine whether the ranges are different
        vary_layers = self.layer_range['min'] != self.layer_range['max']
        vary_hidden_channels = self.hidden_channels_range['min'] != self.hidden_channels_range['max']
        vary_epochs = self.epoch_range['min'] != self.epoch_range['max']

        # Default values in case the range is fixed
        num_layers_fixed = self.layer_range['min']
        hidden_channels_fixed = self.hidden_channels_range['min']
        epochs_fixed = self.epoch_range['min']

        # Loop over activation functions

        #activation_fn = 0


        # Loop over num_layers only if the range is variable
        if vary_layers:
            num_layers_values = range(self.layer_range['min'], self.layer_range['max'] + 1, self.layer_range['step'])
        else:
            num_layers_values = [num_layers_fixed]

        # Loop over hidden_channels only if the range is variable
        if vary_hidden_channels:
            hidden_channels_values = range(self.hidden_channels_range['min'], 
                                            self.hidden_channels_range['max'] + 1, 
                                            self.hidden_channels_range['step'])
        else:
            hidden_channels_values = [hidden_channels_fixed]

        # Loop over epochs only if the range is variable
        if vary_epochs:
            epochs_values = range(self.epoch_range['min'], self.epoch_range['max'] + 1, self.epoch_range['step'])
        else:
            epochs_values = [epochs_fixed]

        # Perform the hyperparameter search
        for num_layers in num_layers_values:
            for hidden_channels in hidden_channels_values:
                for epochs in epochs_values:
                    print(f"Training with {num_layers} layers, {hidden_channels} hidden channels, {epochs} epochs, "
                            f"{self.activation_name} activation, learning rate {self.learning_rate}")
                    
                    # Update model kwargs
                    
                    self.model_kwargs['model_name'] = self.model_name
                    self.model_kwargs['num_layers'] = num_layers
                    self.model_kwargs['num_features'] = self.num_features
                    self.model_kwargs['out_channels'] = self.out_channels
                    self.model_kwargs['hidden_channels'] = hidden_channels
                    self.model_kwargs['activation_name'] = self.activation_name
                    self.model_kwargs['loss_name'] = self.loss_name
                    self.model_kwargs['optimizer_name'] = self.optimizer_name
                    self.model_kwargs['learning_rate'] = self.learning_rate
                    self.model_kwargs['skip'] = self.skip
                    self.model_kwargs['device'] = self.device
                    self.model_kwargs['ppr'] = self.ppr
                    self.model_kwargs['dropout'] = self.dprate
                    
                    print('ACTIVATION NAME:', self.activation_name)

                    # Include num_heads if it's a GAT model
                    if self.model_name == 'GAT':
                        self.model_kwargs['num_heads'] = self.num_heads
                        
                    # Include extra for GPRGNN
                    elif self.model_name == 'GPRGNN' or self.model_name == 'BLOCK_APPNP':
                        self.model_kwargs['ppnp'] = self.ppnp
                        self.model_kwargs['K'] = self.K
                        self.model_kwargs['alpha'] = self.alpha
                        self.model_kwargs['dprate'] = self.dprate
                    
                    self.mw = ModelWrapper(self.data, **self.model_kwargs)
                    

                    if not self.gpp:
                        # Get adjacency matrix & remote adjacency matrix
                        self.adj_matrix, self.rmt_adj_matrix = self.build_adj_mat()
                        
                        # Train the model and get the best train loss
                        best_train_loss = self.train(epochs=epochs, 
                                                    num_layers=num_layers, 
                                                    hidden_channels=hidden_channels)
                    else:
                        pass
                    '''
                        best_train_loss = self.train_gpp(data, 
                                                    optimizer_name=optimizer, 
                                                    loss_name=loss, 
                                                    epochs=epochs, 
                                                    learning_rate=learning_rate, 
                                                    num_layers=num_layers, 
                                                    hidden_channels=hidden_channels)
                    '''
    def train(self, num_layers, hidden_channels, epochs):
        
        # Get the optimizer and loss function based on names
        optimizer = self.mw.optimizer
        loss_fn = self.mw.loss
        
        # get model type:
        model_type = self.model_name

        # Set the model to training mode
        self.mw.model.train()

        # Track the best train loss
        best_train_loss = float('inf')

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.mw.model(self.data.x.to(self.device), self.data.edge_index.to(self.device), ppr_matrix=self.mw.ppr_matrix)
            train_loss = loss_fn(out[self.data.train_mask.to(self.device)], self.data.y[self.data.train_mask].to(self.device))
            train_loss.backward()
            optimizer.step()
            
            # train loss as a float
            train_loss_cpu = train_loss.item()
            
            train_accuracy = self.calculate_accuracy(out[self.data.train_mask.to(self.device)], self.data.y[self.data.train_mask].to(self.device))

            # Update the best train loss
            if train_loss.item() < best_train_loss:
                best_train_loss = train_loss.item()
                
                
            if (epoch+1) % self.save_interval == 0 or epoch+1 < 10:
                # Get hidden representations
                hidden_representations = self.get_hidden_representations()
                #self.model.train()
                
                with torch.no_grad():
                # make the labels one-hot
                    out_mse = self.mw.model.forward_2(self.data.x.to(self.device), self.data.edge_index.to(self.device), self.mw.ppr_matrix)
                    # Apply softmax to get probabilities
                    out_mse = F.softmax(out_mse, dim=1)
                    one_hot_labels = F.one_hot(self.data.y[self.data.train_mask].to(self.device), num_classes=out_mse.shape[1])
                    train_mse_loss = self.mw.mse_loss(out_mse[self.data.train_mask.to(self.device)], one_hot_labels.float())
                    train_mse_loss = train_mse_loss.item()
                    
                test_loss, test_mse_loss, test_accuracy = self.test(self.data, out_mse, loss_name=self.loss_name)
                
                # Calculate the MadValue
                mad_value = self.measure.get_mad_value(hidden_representations, self.adj_matrix, distance_metric='cosine', digt_num=4, target_idx=None)
                
                # Calculate the Mad_remote
                mad_rmt = self.measure.get_mad_value(hidden_representations, self.rmt_adj_matrix, distance_metric='cosine', digt_num=4, target_idx=None)
                
                # Calculate the MadGap
                madgap_value = round(float(mad_rmt - mad_value), 4)
                
                
                # Calculate the Dirichlet Energy
                dirichlet_energy = self.measure.get_dirichlet_energy(hidden_representations, self.adj_matrix)
                dirichlet_energy = round(float(dirichlet_energy), 4)
                
                # Save the results to a CSV file
                results = [
                    {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                        "train_loss": train_loss_cpu, "train_mse_loss": train_mse_loss, "train_accuracy": train_accuracy, 
                        "test_loss": test_loss, "test_mse_loss": test_mse_loss, "test_accuracy": test_accuracy, 
                        "mad_value": mad_value, "mad_gap": madgap_value, "dirichlet_energy": dirichlet_energy},
                ]
                self.save_training_results(results)
                
        return best_train_loss
    
    def test(self, data, hidden_representations, loss_name="cross_entropy"):
        # Get the loss function based on name
        loss_fn = self.mw.loss

        # Set the model to evaluation mode
        self.mw.model.eval()

        # Perform the forward pass for predictions
        with torch.no_grad():
            out = self.mw.model(data.x.to(self.device), data.edge_index.to(self.device), ppr_matrix=self.mw.ppr_matrix)
            test_loss = loss_fn(out[data.test_mask.to(self.device)], data.y[data.test_mask].to(self.device))
            test_accuracy = self.calculate_accuracy(out[data.test_mask.to(self.device)], data.y[data.test_mask].to(self.device))
            
            # make the labels one-hot
            one_hot_labels = F.one_hot(data.y[data.test_mask].to(self.device), num_classes=hidden_representations.shape[1])
            test_mse_loss = self.mw.mse_loss(hidden_representations[data.test_mask.to(self.device)], one_hot_labels.float())

        return test_loss.item(), test_mse_loss.item(), test_accuracy
    
    def calculate_accuracy(self, logits, labels):
        _, preds = torch.max(logits, dim=1)
        correct = (preds == labels).sum().item()
        accuracy = correct / len(labels)
        return accuracy
    
    def build_adj_mat(self):
        hidden_representations = self.get_hidden_representations()
        
        # Create adjacency matrix based on graph structure
        adjacency_matrix = torch.zeros((hidden_representations.shape[0], hidden_representations.shape[0]))
        adjacency_matrix[self.data.edge_index[0], self.data.edge_index[1]] = 1
        
        
        rmt_adjacency_matrix = torch.ones((hidden_representations.shape[0], hidden_representations.shape[0])) - adjacency_matrix
        
        adjacency_matrix = adjacency_matrix.cpu().numpy()
        rmt_adjacency_matrix = rmt_adjacency_matrix.cpu().numpy()
        
        return adjacency_matrix, rmt_adjacency_matrix
    
    def get_hidden_representations(self):
        self.mw.model.eval()
        with torch.no_grad():
            #hidden_representations = model(data).detach().cpu().numpy()
            try:
                hidden_representations = self.mw.model.forward_2(self.data.x.to(self.device), self.data.edge_index.to(self.device), self.mw.ppr_matrix).detach().cpu().numpy()
                
            except:
                hidden_representations = self.mw.model(self.data.x.to(self.device), self.data.edge_index.to(self.device), self.mw.ppr_matrix).detach().cpu().numpy()
        
        return hidden_representations
    
    def save_training_results(self, results):
        file_exists = os.path.isfile(self.output_path)
        
        with open(self.output_path, 'a', newline='') as csvfile:
            fieldnames = ['model_type', 'layers', "hidden_channels", "epochs", 
                        'train_loss', 'train_mse_loss', 'train_accuracy', 
                        'test_loss', 'test_mse_loss', 'test_accuracy', 
                        'mad_value', 'mad_gap', 'dirichlet_energy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            for result in results:
                writer.writerow(result)
