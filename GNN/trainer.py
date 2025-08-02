from model_wrapper import ModelWrapper
from utils.oversmooth import OversmoothMeasure

import numpy as np
import torch
import torch.nn.functional as F
import os
import csv
import pandas as pd

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
                    trial_num=1,
                    use_madreg=False,
                    lambda_reg=0.1,
                    **kwargs):
        
        
        self.data = data
        self.model_name = model_name
        self.num_features = num_features
        self.out_channels = out_channels
        self.device = device
        self.output_path = output_path
        self.evaluator=evaluator
        self.trial_num = trial_num
        
        self.gpp = gpp
        self.skip = skip
        self.ppr = ppr
        
        # MADReg settings
        self.use_madreg = use_madreg
        self.lambda_reg = lambda_reg
        
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

        # Progressive width specific parameters
        self.progressive_width = hyperparams.get("progressive_width", True)
        self.base_lr_scaling = hyperparams.get("base_lr_scaling", True)
        self.gradient_clipping = hyperparams.get("gradient_clipping", 1.0)
        self.warmup_epochs_ratio = hyperparams.get("warmup_epochs_ratio", 0.1)
        
        # CONFIGURABLE LOGGING PARAMETERS
        self.start_epochs_to_save = hyperparams.get("start_epochs_to_save", 10)
        self.train_log_frequency = hyperparams.get("train_log_frequency", 5)
        self.mad_calculation_frequency = hyperparams.get("mad_calculation_frequency", 10)
        
        # Adjust learning rate ranges for progressive width models
        if self.progressive_width:
            self.learning_rate_range = hyperparams.get("learning_rate_range", {
                "min": 0.001, "max": 0.01, "values": [0.001, 0.005, 0.01]
            })

        # Store the arguments needed for model initialization
        self.model_kwargs = kwargs
        
        self.measure = OversmoothMeasure()    
        
        # Create trial-specific checkpoint directory
        self.checkpoint_dir = os.path.dirname(self.output_path)
        self.checkpoint_base = os.path.join(self.checkpoint_dir, "checkpoints")
        os.makedirs(self.checkpoint_base, exist_ok=True)
        
        print(f"Trainer initialized for trial {self.trial_num}")
        print(f"MADReg enabled: {self.use_madreg}")
        if self.use_madreg:
            print(f"  - Lambda regularization: {self.lambda_reg}")
        print(f"Progressive width enabled: {self.progressive_width}")
        if self.progressive_width:
            print(f"  - Base K range: {self.hidden_channels_range}")
            print(f"  - Layer range: {self.layer_range}")
            print(f"  - Adaptive learning rate: {self.base_lr_scaling}")
            print(f"  - Gradient clipping: {self.gradient_clipping}")
        
        # Print logging configuration
        print(f"Logging configuration:")
        print(f"  - Start epochs to save: {self.start_epochs_to_save}")
        print(f"  - Train log frequency: every {self.train_log_frequency} epochs")
        print(f"  - MAD calculation frequency: every {self.mad_calculation_frequency} epochs")
        print(f"  - Comprehensive log interval: every {self.save_interval} epochs")
        
        print(f"Output path: {self.output_path}")
        print(f"Train-only output path: {self.get_train_only_output_path()}")
        print(f"Checkpoint directory: {self.checkpoint_base}")
        
    def get_train_only_output_path(self):
        """
        Generate a separate output path for training-only metrics
        """
        base_dir = os.path.dirname(self.output_path)
        base_filename = os.path.basename(self.output_path)
        
        # Replace .csv with _train_only.csv
        if base_filename.endswith('.csv'):
            train_only_filename = base_filename.replace('.csv', '_train_only.csv')
        else:
            train_only_filename = base_filename + '_train_only.csv'
        
        return os.path.join(base_dir, train_only_filename)

    def save_train_only_results(self, results):
        """
        Save training-only results to a separate CSV file
        """
        train_only_path = self.get_train_only_output_path()
        file_exists = os.path.isfile(train_only_path)
        
        with open(train_only_path, 'a', newline='') as csvfile:
            if self.use_madreg:
                fieldnames = ['model_type', 'layers', 'hidden_channels', 'epochs', 
                            'train_loss', 'total_loss', 'madreg_loss', 'train_accuracy', 
                            'mad_value', 'mad_gap', 'dirichlet_energy', 'trial',
                            'learning_rate', 'lambda_reg']
            else:
                fieldnames = ['model_type', 'layers', 'hidden_channels', 'epochs', 
                            'train_loss', 'train_accuracy', 
                            'mad_value', 'mad_gap', 'dirichlet_energy', 'trial',
                            'learning_rate']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            for result in results:
                writer.writerow(result)
        
    def calculate_madreg_loss(self, hidden_representations):
        """
        Calculate MADReg regularization loss based on MADGap
        """
        # Calculate MAD for neighboring nodes (small topological distance)
        mad_neb = self.measure.get_mad_value(hidden_representations, self.adj_matrix, 
                                            distance_metric='cosine', digt_num=4, target_idx=None)
        
        # Calculate MAD for remote nodes (large topological distance) 
        mad_rmt = self.measure.get_mad_value(hidden_representations, self.rmt_adj_matrix, 
                                            distance_metric='cosine', digt_num=4, target_idx=None)
        
        # Calculate MADGap
        madgap_value = mad_rmt - mad_neb
        
        # MADReg loss is negative MADGap (we want to maximize MADGap, so minimize -MADGap)
        madreg_loss = -self.lambda_reg * madgap_value
        
        return madreg_loss, madgap_value
        
    def calculate_model_complexity(self, num_layers, hidden_channels):
        """
        Calculate approximate model complexity for progressive width models.
        """
        if not self.progressive_width:
            # Standard model: all layers have same width
            total_params = 0
            for i in range(num_layers):
                if i == 0:
                    layer_params = self.num_features * hidden_channels
                elif i == num_layers - 1:
                    layer_params = hidden_channels * self.out_channels
                else:
                    layer_params = hidden_channels * hidden_channels
                total_params += layer_params
        else:
            # Progressive width: K, 2K, 3K, ..., nK - FIXED
            total_params = 0
            for i in range(num_layers):
                if i == 0:
                    layer_params = self.num_features * hidden_channels
                elif i == num_layers - 1:
                    layer_params = hidden_channels * i * self.out_channels
                else:
                    layer_params = hidden_channels * i * hidden_channels * (i + 1)
                total_params += layer_params
        
        return total_params
    
    def get_adaptive_learning_rate(self, num_layers, hidden_channels):
        """
        Calculate adaptive learning rate based on model complexity.
        """
        if not self.base_lr_scaling or not self.progressive_width:
            return self.learning_rate
        
        base_lr = self.learning_rate
        model_complexity = self.calculate_model_complexity(num_layers, hidden_channels)
        
        # Calculate baseline complexity (2-layer standard model)
        baseline_complexity = self.num_features * hidden_channels + hidden_channels * self.out_channels
        
        # Scale learning rate inversely with relative model complexity
        complexity_ratio = model_complexity / baseline_complexity
        scaling_factor = 1.0 / np.sqrt(complexity_ratio)
        
        adaptive_lr = base_lr * scaling_factor
        
        # Clamp learning rate to reasonable bounds
        adaptive_lr = np.clip(adaptive_lr, 0.0001, 0.1)
        
        print(f"Model complexity: {model_complexity:,} params (baseline: {baseline_complexity:,}), "
              f"Base LR: {base_lr:.4f}, Adaptive LR: {adaptive_lr:.4f}")
        
        return adaptive_lr
    
    def setup_progressive_width_model(self, num_layers, hidden_channels):
        """
        Setup model with progressive width specific configurations.
        """
        # Calculate adaptive learning rate
        adaptive_lr = self.get_adaptive_learning_rate(num_layers, hidden_channels)
        
        # Update model kwargs with adaptive learning rate
        self.model_kwargs['learning_rate'] = adaptive_lr
        
        # Add progressive width specific parameters
        self.model_kwargs['progressive_width'] = self.progressive_width
        
        print(f"Trial {self.trial_num}: Progressive width model setup - "
              f"L{num_layers}, base_K={hidden_channels}, adaptive_lr={adaptive_lr:.4f}")
        
        return adaptive_lr
        
    def get_checkpoint_path(self, num_layers, hidden_channels):
        """Generate trial-specific checkpoint path for a specific configuration"""
        madreg_suffix = f"_madreg{self.lambda_reg}" if self.use_madreg else ""
        return os.path.join(self.checkpoint_base, 
                           f"{self.model_name}_L{num_layers}_H{hidden_channels}_trial{self.trial_num}{madreg_suffix}.pth")
    
    def save_checkpoint(self, model, optimizer, epoch, num_layers, hidden_channels, 
                       train_loss, best_train_loss):
        """Save model checkpoint with trial-specific naming"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'best_train_loss': best_train_loss,
            'num_layers': num_layers,
            'hidden_channels': hidden_channels,
            'trial_num': self.trial_num,
            'model_kwargs': self.model_kwargs,
            'progressive_width': self.progressive_width,
            'use_madreg': self.use_madreg,
            'lambda_reg': self.lambda_reg
        }
        
        checkpoint_path = self.get_checkpoint_path(num_layers, hidden_channels)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    def load_checkpoint(self, num_layers, hidden_channels):
        """Load trial-specific model checkpoint if it exists"""
        checkpoint_path = self.get_checkpoint_path(num_layers, hidden_channels)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            return checkpoint
        return None
    
    def clean_incomplete_training_logs(self):
        """
        Clean the log file by removing incomplete training entries.
        """
        if not os.path.exists(self.output_path):
            print(f"No existing log file found at {self.output_path}")
            return
        
        try:
            df = pd.read_csv(self.output_path)
            if df.empty:
                return
            
            max_epochs = self.epoch_range['max']
            original_length = len(df)
            
            # Group by configuration and find which ones are incomplete
            grouped = df.groupby(['layers', 'hidden_channels'])
            configurations_to_keep = []
            
            for (layers, hidden_channels), group in grouped:
                max_epoch_completed = group['epochs'].max()
                
                if max_epoch_completed >= max_epochs:
                    configurations_to_keep.append((layers, hidden_channels))
                    print(f"Trial {self.trial_num}: Configuration L{layers}_H{hidden_channels} is complete ({max_epoch_completed} epochs) - keeping entries")
                else:
                    print(f"Trial {self.trial_num}: Configuration L{layers}_H{hidden_channels} is incomplete ({max_epoch_completed} epochs) - removing entries")
                    checkpoint_path = self.get_checkpoint_path(layers, hidden_channels)
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                        print(f"Removed incomplete checkpoint: {checkpoint_path}")
            
            # Filter dataframe to keep only complete configurations
            if configurations_to_keep:
                mask = df.apply(lambda row: (row['layers'], row['hidden_channels']) in configurations_to_keep, axis=1)
                cleaned_df = df[mask]
            else:
                cleaned_df = df.iloc[0:0].copy()
            
            cleaned_df.to_csv(self.output_path, index=False)
            
            removed_entries = original_length - len(cleaned_df)
            if removed_entries > 0:
                print(f"Trial {self.trial_num}: Cleaned log file: removed {removed_entries} incomplete entries, kept {len(cleaned_df)} complete entries")
            else:
                print(f"Trial {self.trial_num}: Log file is already clean: all {len(cleaned_df)} entries represent complete training runs")
                
        except Exception as e:
            print(f"Error cleaning log file {self.output_path}: {e}")

    def get_training_progress(self):
        """
        Check existing training progress from CSV log file.
        """
        if not os.path.exists(self.output_path):
            print(f"Trial {self.trial_num}: No existing log file found at {self.output_path}")
            return {}
        
        try:
            df = pd.read_csv(self.output_path)
            if df.empty:
                return {}
            
            progress = {}
            max_epochs = self.epoch_range['max']
            
            grouped = df.groupby(['layers', 'hidden_channels'])
            
            for (layers, hidden_channels), group in grouped:
                max_epoch_completed = group['epochs'].max()
                progress[(layers, hidden_channels)] = {
                    'max_epoch_completed': max_epoch_completed,
                    'is_complete': max_epoch_completed >= max_epochs,
                    'last_train_loss': group.loc[group['epochs'].idxmax(), 'train_loss']
                }
            
            return progress
            
        except Exception as e:
            print(f"Error reading progress from {self.output_path}: {e}")
            return {}
    
    def should_skip_configuration(self, num_layers, hidden_channels, progress):
        """
        Determine if a configuration should be skipped based on existing progress
        """
        config_key = (num_layers, hidden_channels)
        
        if config_key in progress:
            config_progress = progress[config_key]
            if config_progress['is_complete']:
                print(f"Trial {self.trial_num}: Configuration L{num_layers}_H{hidden_channels} already completed "
                      f"({config_progress['max_epoch_completed']} epochs). Skipping.")
                return True
        
        return False
        
        
    def hyperparameter_train(self):
        """
        Enhanced hyperparameter search with optional MADReg regularization
        """
        # First, clean any incomplete training logs
        self.clean_incomplete_training_logs()
        
        # Check existing progress after cleaning
        progress = self.get_training_progress()
        
        # Determine whether the ranges are different
        vary_layers = self.layer_range['min'] != self.layer_range['max']
        vary_hidden_channels = self.hidden_channels_range['min'] != self.hidden_channels_range['max']
        vary_epochs = self.epoch_range['min'] != self.epoch_range['max']

        # Default values in case the range is fixed
        num_layers_fixed = self.layer_range['min']
        hidden_channels_fixed = self.hidden_channels_range['min']
        epochs_fixed = self.epoch_range['min']

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

        # Enhanced logging
        print(f"Trial {self.trial_num}: Training with {'MADReg enabled' if self.use_madreg else 'standard training'}")
        if self.use_madreg:
            print(f"  - Lambda regularization: {self.lambda_reg}")
        if self.progressive_width:
            print(f"  - Progressive width enabled: {self.progressive_width}")
            print(f"  - Base K range: {self.hidden_channels_range}")
            print(f"  - Layer range: {self.layer_range}")
            print(f"  - Adaptive learning rate: {self.base_lr_scaling}")
            print(f"  - Gradient clipping: {self.gradient_clipping}")

        # Perform the hyperparameter search
        for num_layers in num_layers_values:
            for hidden_channels in hidden_channels_values:
                for epochs in epochs_values:
                    
                    # Check if this configuration should be skipped (already completed)
                    if self.should_skip_configuration(num_layers, hidden_channels, progress):
                        continue
                    
                    # Calculate expected model complexity for progressive width
                    if self.progressive_width:
                        model_complexity = self.calculate_model_complexity(num_layers, hidden_channels)
                        print(f"Trial {self.trial_num}: Training {'MADReg' if self.use_madreg else 'standard'} model - "
                              f"L{num_layers}, base_K={hidden_channels}, ~{model_complexity:,} params")
                    
                    method_str = "MADReg" if self.use_madreg else "standard"
                    print(f"Trial {self.trial_num}: Training {method_str} with {num_layers} layers, {hidden_channels} hidden channels, {epochs} epochs")
                    if self.use_madreg:
                        print(f"  MADReg lambda: {self.lambda_reg}")
                    
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
                    
                    # Create initial model wrapper
                    self.mw = ModelWrapper(self.data, **self.model_kwargs)
                    

                    if not self.gpp:
                        # Get adjacency matrix & remote adjacency matrix
                        self.adj_matrix, self.rmt_adj_matrix = self.build_adj_mat()
                        
                        # Train the model
                        best_train_loss = self.train(epochs=epochs, 
                                                    num_layers=num_layers, 
                                                    hidden_channels=hidden_channels)
                    else:
                        best_train_loss = self.train_gpp(self.data, 
                                                         epochs=epochs, 
                                                         num_layers=num_layers, 
                                                         hidden_channels=hidden_channels)

    def train(self, num_layers, hidden_channels, epochs):
        
        # Setup progressive width specific configurations
        if self.progressive_width:
            adaptive_lr = self.setup_progressive_width_model(num_layers, hidden_channels)
            self.mw = ModelWrapper(self.data, **self.model_kwargs)
        
        # Get the optimizer and loss function based on names
        optimizer = self.mw.optimizer
        loss_fn = self.mw.loss
        
        # get model type:
        model_type = self.model_name
        if self.use_madreg:
            model_type += "_MADReg"

        # Set the model to training mode
        self.mw.model.train()

        # Track the best train loss
        best_train_loss = float('inf')
        
        # Store the train-only results for later saving
        train_only_results = []
        
        # Progressive width models may need warmup
        warmup_epochs = int(epochs * self.warmup_epochs_ratio) if self.progressive_width else 0
        warmup_epochs = max(1, min(warmup_epochs, 20))

        #print(f"Starting optimized training with configurable logging:")
        #print(f"  - Start epochs to save: {self.start_epochs_to_save}")
        #print(f"  - Train log frequency: every {self.train_log_frequency} epochs") 
        #print(f"  - MAD calculation frequency: every {self.mad_calculation_frequency} epochs")
        #print(f"  - Train-only CSV: {self.get_train_only_output_path()}")

        # Pre-move data to device once (avoid repeated .to() calls)
        x_device = self.data.x.to(self.device)
        edge_index_device = self.data.edge_index.to(self.device)
        train_mask_device = self.data.train_mask.to(self.device)
        y_train_device = self.data.y[self.data.train_mask].to(self.device)
        
        # Pre-compute adjacency matrices once (avoid recomputation every epoch)
        if not hasattr(self, 'adj_matrix') or not hasattr(self, 'rmt_adj_matrix'):
            #print("Computing adjacency matrices once...")
            self.adj_matrix, self.rmt_adj_matrix = self.build_adj_mat()
        
        # Training loop
        for epoch in range(epochs):
            
            # Learning rate warmup for progressive width models
            if self.progressive_width and epoch < warmup_epochs:
                if hasattr(self, 'adaptive_lr'):
                    warmup_lr = self.model_kwargs['learning_rate'] * (epoch + 1) / warmup_epochs
                else:
                    warmup_lr = self.learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            elif self.progressive_width and epoch == warmup_epochs:
                full_lr = self.model_kwargs.get('learning_rate', self.learning_rate)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = full_lr
                    
            optimizer.zero_grad()
            
            # SINGLE FORWARD PASS - used for both training and logging
            out = self.mw.model(x_device, edge_index_device, ppr_matrix=self.mw.ppr_matrix)
            
            # Standard cross-entropy loss (using pre-moved tensors)
            train_loss = loss_fn(out[train_mask_device], y_train_device)
            
            # Calculate training accuracy from the same forward pass
            train_accuracy = self.calculate_accuracy(out[train_mask_device], y_train_device)
            
            # Initialize total loss and MADReg components
            total_loss = train_loss
            madreg_loss_value = 0.0
            madgap_value = 0.0
            
            # Add MADReg regularization if enabled
            if self.use_madreg:
                with torch.no_grad():
                    hidden_representations = out.detach().cpu().numpy()
                
                madreg_loss, madgap_value = self.calculate_madreg_loss(hidden_representations)
                total_loss = train_loss + madreg_loss
                madreg_loss_value = madreg_loss.item() if hasattr(madreg_loss, 'item') else madreg_loss
            
            total_loss.backward()
            
            # Gradient clipping for stability
            if self.progressive_width and self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.mw.model.parameters(), self.gradient_clipping)
            
            optimizer.step()
            
            # train loss as a float
            train_loss_cpu = train_loss.item()
            total_loss_cpu = total_loss.item()

            # Update the best train loss
            if total_loss.item() < best_train_loss:
                best_train_loss = total_loss.item()
            
            # TRAINING-ONLY LOGGING WITH CONFIGURABLE FREQUENCIES
            # MAD calculations using configurable frequency
            compute_mad_values = (epoch + 1) % self.mad_calculation_frequency == 0 or (epoch + 1) <= self.start_epochs_to_save or (epoch + 1) == epochs
            
            if compute_mad_values:
                # Get hidden representations from the SAME forward pass if not already done for MADReg
                if not self.use_madreg:
                    with torch.no_grad():
                        hidden_representations = out.detach().cpu().numpy()
                
                # Calculate MAD values
                mad_value = self.measure.get_mad_value(hidden_representations, self.adj_matrix, distance_metric='cosine', digt_num=4, target_idx=None)
                mad_rmt = self.measure.get_mad_value(hidden_representations, self.rmt_adj_matrix, distance_metric='cosine', digt_num=4, target_idx=None)
                
                # Calculate MADGap if not already calculated for MADReg
                if not self.use_madreg:
                    madgap_value = round(float(mad_rmt - mad_value), 4)
                
                # Calculate Dirichlet Energy
                dirichlet_energy = self.measure.get_dirichlet_energy(hidden_representations, self.adj_matrix)
                dirichlet_energy = round(float(dirichlet_energy), 4)
            else:
                # Use cached/zero values for non-critical epochs
                mad_value = 0.0
                madgap_value = 0.0
                dirichlet_energy = 0.0
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save training-only results using configurable frequency
            should_log_train_only = (epoch + 1) % self.train_log_frequency == 0 or (epoch + 1) <= self.start_epochs_to_save or (epoch + 1) == epochs
            
            if should_log_train_only:
                if self.use_madreg:
                    
                    train_only_results.append({
                        "model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                        "train_loss": train_loss_cpu, "total_loss": total_loss_cpu, "madreg_loss": madreg_loss_value,
                        "train_accuracy": train_accuracy, "mad_value": mad_value, "mad_gap": madgap_value, 
                        "dirichlet_energy": dirichlet_energy, "trial": self.trial_num, 
                        "learning_rate": current_lr, "lambda_reg": self.lambda_reg
                    })
                    
                    '''
                    train_only_results = [
                        {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                            "train_loss": train_loss_cpu, "total_loss": total_loss_cpu, "madreg_loss": madreg_loss_value,
                            "train_accuracy": train_accuracy, "mad_value": mad_value, "mad_gap": madgap_value, 
                            "dirichlet_energy": dirichlet_energy, "trial": self.trial_num, 
                            "learning_rate": current_lr, "lambda_reg": self.lambda_reg}
                    ]
                    '''
                else:
                    
                    train_only_results.append({
                        "model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                        "train_loss": train_loss_cpu, "train_accuracy": train_accuracy, 
                        "mad_value": mad_value, "mad_gap": madgap_value, "dirichlet_energy": dirichlet_energy, 
                        "trial": self.trial_num, "learning_rate": current_lr
                    })
                    
                    '''
                    train_only_results = [
                        {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                            "train_loss": train_loss_cpu, "train_accuracy": train_accuracy, 
                            "mad_value": mad_value, "mad_gap": madgap_value, "dirichlet_energy": dirichlet_energy, 
                            "trial": self.trial_num, "learning_rate": current_lr}
                    ]
                    '''
                
                #self.save_train_only_results(train_only_results)
                
            # ORIGINAL COMPREHENSIVE LOGGING (EVERY save_interval EPOCHS)
            if (epoch+1) % self.save_interval == 0 or (epoch+1) <= self.start_epochs_to_save:
                with torch.no_grad():
                    # Only NOW do we compute the more expensive forward_2 for MSE loss
                    try:
                        out_mse = self.mw.model.forward_2(x_device, edge_index_device, self.mw.ppr_matrix)
                    except:
                        out_mse = out  # Fallback to regular output if forward_2 doesn't exist
                    
                    # Apply softmax to get probabilities
                    out_mse = F.softmax(out_mse, dim=1)
                    one_hot_labels = F.one_hot(y_train_device, num_classes=out_mse.shape[1])
                    train_mse_loss = self.mw.mse_loss(out_mse[train_mask_device], one_hot_labels.float())
                    train_mse_loss = train_mse_loss.item()
                    
                # Expensive test evaluation only at save intervals
                test_loss, test_mse_loss, test_accuracy = self.test(self.data, out_mse, loss_name=self.loss_name)
                
                # Calculate actual model complexity for logging (cached)
                if not hasattr(self, '_cached_model_complexity'):
                    self._cached_model_complexity = self.calculate_model_complexity(num_layers, hidden_channels)
                model_complexity = self._cached_model_complexity
                
                # Ensure we have MAD values for comprehensive logging
                if not compute_mad_values:
                    with torch.no_grad():
                        if not self.use_madreg:
                            hidden_representations = out.detach().cpu().numpy()
                    mad_value = self.measure.get_mad_value(hidden_representations, self.adj_matrix, distance_metric='cosine', digt_num=4, target_idx=None)
                    mad_rmt = self.measure.get_mad_value(hidden_representations, self.rmt_adj_matrix, distance_metric='cosine', digt_num=4, target_idx=None)
                    if not self.use_madreg:
                        madgap_value = round(float(mad_rmt - mad_value), 4)
                    dirichlet_energy = self.measure.get_dirichlet_energy(hidden_representations, self.adj_matrix)
                    dirichlet_energy = round(float(dirichlet_energy), 4)
                
                # Save the comprehensive results to the main CSV file
                if self.use_madreg:
                    results = [
                        {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                            "train_loss": train_loss_cpu, "total_loss": total_loss_cpu, "madreg_loss": madreg_loss_value,
                            "train_mse_loss": train_mse_loss, "train_accuracy": train_accuracy, 
                            "test_loss": test_loss, "test_mse_loss": test_mse_loss, "test_accuracy": test_accuracy, 
                            "mad_value": mad_value, "mad_gap": madgap_value, "dirichlet_energy": dirichlet_energy, 
                            "trial": self.trial_num, "learning_rate": current_lr, "progressive_width": self.progressive_width,
                            "model_complexity": model_complexity, "lambda_reg": self.lambda_reg},
                    ]
                    self.save_training_results_madreg(results)
                else:
                    results = [
                        {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                            "train_loss": train_loss_cpu, "train_mse_loss": train_mse_loss, "train_accuracy": train_accuracy, 
                            "test_loss": test_loss, "test_mse_loss": test_mse_loss, "test_accuracy": test_accuracy, 
                            "mad_value": mad_value, "mad_gap": madgap_value, "dirichlet_energy": dirichlet_energy, 
                            "trial": self.trial_num, "learning_rate": current_lr, "progressive_width": self.progressive_width,
                            "model_complexity": model_complexity},
                    ]
                    self.save_training_results(results)
                
                # CHECKPOINT SAVING COMMENTED OUT
                # if (epoch + 1) % (self.save_interval * 5) == 0:
                #     self.save_checkpoint(self.mw.model, optimizer, epoch + 1, num_layers, 
                #                        hidden_channels, train_loss_cpu, best_train_loss)
        
        # FINAL CHECKPOINT SAVING COMMENTED OUT
        # self.save_checkpoint(self.mw.model, optimizer, epochs, num_layers, 
        #                    hidden_channels, train_loss_cpu, best_train_loss)
                
        # Save the final train-only results to CSV
        self.save_train_only_results(train_only_results)
                
        #print(f"Training completed. Train-only metrics saved to: {self.get_train_only_output_path()}")
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
    
    def train_gpp(self, data, epochs=100, num_layers=2, hidden_channels=16):
        
        # Setup progressive width specific configurations for GPP
        if self.progressive_width:
            adaptive_lr = self.setup_progressive_width_model(num_layers, hidden_channels)
            self.mw = ModelWrapper(data, **self.model_kwargs)
        
        # Get the optimizer and loss function based on names
        optimizer = self.mw.optimizer
        loss_fn = self.mw.loss
        
        # get model type:
        model_type = self.model_name
        if self.use_madreg:
            model_type += "_MADReg"

        # Set the model to training mode
        self.mw.model.train()
        
        train_loader, valid_loader, test_loader = data
        
        # Progressive width models may need warmup
        warmup_epochs = int(epochs * self.warmup_epochs_ratio) if self.progressive_width else 0
        warmup_epochs = max(1, min(warmup_epochs, 20))
        
        print(f"Starting optimized GPP training with configurable logging:")
        print(f"  - Start epochs to save: {self.start_epochs_to_save}")
        print(f"  - Train log frequency: every {self.train_log_frequency} epochs")
        print(f"  - Train-only CSV: {self.get_train_only_output_path()}")
        
        # Training loop
        for epoch in range(epochs):
            
            # Learning rate warmup for progressive width models
            if self.progressive_width and epoch < warmup_epochs:
                if hasattr(self, 'adaptive_lr'):
                    warmup_lr = self.model_kwargs['learning_rate'] * (epoch + 1) / warmup_epochs
                else:
                    warmup_lr = self.learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            elif self.progressive_width and epoch == warmup_epochs:
                full_lr = self.model_kwargs.get('learning_rate', self.learning_rate)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = full_lr
            
            running_loss = 0.0
            running_mse_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            for batch in train_loader:
                
                batch = batch.to(self.device, non_blocking=True)  # Non-blocking transfer for speed

                if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                    continue  # Skip invalid batches more efficiently
                
                out = self.mw.model(batch)
                optimizer.zero_grad()

                loss = loss_fn(out, batch.y)
                
                # Note: MADReg for GPP models would need additional implementation
                if self.use_madreg:
                    print("Warning: MADReg for GPP models not fully implemented yet")
                
                loss.backward()
                
                # Gradient clipping for progressive width models
                if self.progressive_width and self.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.mw.model.parameters(), self.gradient_clipping)
                
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate training accuracy for this batch (optimized)
                with torch.no_grad():
                    pred = out.argmax(dim=1)
                    total_correct += (pred == batch.y).sum().item()
                    total_samples += batch.y.size(0)
                    
                    # Create one-hot labels for MSE loss (only if needed for logging)
                    if (epoch + 1) % self.mad_calculation_frequency == 0 or (epoch + 1) <= self.start_epochs_to_save:
                        one_hot_labels = F.one_hot(batch.y, num_classes=out.shape[1])
                        # Apply softmax to get probabilities
                        out_mse = F.softmax(out.to(torch.float32), dim=1)
                        train_mse_loss = self.mw.mse_loss(out_mse, one_hot_labels.float())
                        train_mse_loss = train_mse_loss.item()
                        running_mse_loss += train_mse_loss
                    
            # Calculate epoch averages
            epoch_loss = running_loss / len(train_loader)
            epoch_mse_loss = running_mse_loss / len(train_loader) if running_mse_loss > 0 else 0.0
            epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            
            # Get current learning rate (cached)
            current_lr = optimizer.param_groups[0]['lr']
            
            # TRAINING-ONLY LOGGING using configurable frequency
            should_log_train_only = (epoch + 1) % self.train_log_frequency == 0 or (epoch + 1) <= self.start_epochs_to_save or (epoch + 1) == epochs
            
            if should_log_train_only:
                if self.use_madreg:
                    train_only_results = [
                        {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                            "train_loss": round(float(epoch_loss), 6), "total_loss": round(float(epoch_loss), 6), "madreg_loss": 0.0,
                            "train_accuracy": round(float(epoch_accuracy), 6), 
                            "mad_value": 0, "mad_gap": 0, "dirichlet_energy": 0, 
                            "trial": self.trial_num, "learning_rate": current_lr, "lambda_reg": self.lambda_reg}
                    ]
                else:
                    train_only_results = [
                        {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                            "train_loss": round(float(epoch_loss), 6), "train_accuracy": round(float(epoch_accuracy), 6), 
                            "mad_value": 0, "mad_gap": 0, "dirichlet_energy": 0, 
                            "trial": self.trial_num, "learning_rate": current_lr}
                    ]
                
                self.save_train_only_results(train_only_results)
            
            # ORIGINAL COMPREHENSIVE LOGGING (EVERY save_interval EPOCHS)
            if (epoch+1) % self.save_interval == 0 or (epoch+1) <= self.start_epochs_to_save:
                
                train_accuracy, _, _ = self.eval(train_loader, loss_fn)
                
                test_accuracy, test_loss, test_loss_mse = self.eval(test_loader, loss_fn, test=True)
                
                # Cache model complexity
                if not hasattr(self, '_cached_model_complexity'):
                    self._cached_model_complexity = self.calculate_model_complexity(num_layers, hidden_channels)
                model_complexity = self._cached_model_complexity
                
                if self.use_madreg:
                    results = [
                        {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                            "train_loss": round(float(epoch_loss), 6), "total_loss": round(float(epoch_loss), 6), "madreg_loss": 0.0,
                            "train_mse_loss": round(float(epoch_mse_loss), 6), "train_accuracy": round(float(train_accuracy), 6), 
                            "test_loss": round(float(test_loss), 6), "test_mse_loss": round(float(test_loss_mse), 6), "test_accuracy": round(float(test_accuracy), 6), 
                            "mad_value": 0, "mad_gap": 0, "dirichlet_energy": 0, "trial": self.trial_num,
                            "learning_rate": current_lr, "progressive_width": self.progressive_width,
                            "model_complexity": model_complexity, "lambda_reg": self.lambda_reg},
                    ]
                    self.save_training_results_madreg(results)
                else:
                    results = [
                        {"model_type": model_type, "layers": num_layers, "hidden_channels": hidden_channels, "epochs": epoch+1,
                            "train_loss": round(float(epoch_loss), 6), "train_mse_loss": round(float(epoch_mse_loss), 6), "train_accuracy": round(float(train_accuracy), 6), 
                            "test_loss": round(float(test_loss), 6), "test_mse_loss": round(float(test_loss_mse), 6), "test_accuracy": round(float(test_accuracy), 6), 
                            "mad_value": 0, "mad_gap": 0, "dirichlet_energy": 0, "trial": self.trial_num,
                            "learning_rate": current_lr, "progressive_width": self.progressive_width,
                            "model_complexity": model_complexity},
                    ]
                    self.save_training_results(results)
                
                # CHECKPOINT SAVING COMMENTED OUT
                # if (epoch + 1) % (self.save_interval * 5) == 0:
                #     self.save_checkpoint(self.mw.model, optimizer, epoch + 1, num_layers, 
                #                        hidden_channels, epoch_loss, 0)
        
        # FINAL CHECKPOINT SAVING COMMENTED OUT
        # self.save_checkpoint(self.mw.model, optimizer, epochs, num_layers, 
        #                    hidden_channels, epoch_loss, 0)
            
        print(f"GPP Training completed. Train-only metrics saved to: {self.get_train_only_output_path()}")
        return None
    
    def eval(self, loader, loss_fn, test=False):
        self.mw.model.eval()

        correct = 0
        running_loss = 0.0
        running_mse_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                if batch.x.shape[0] == 1:
                    pass
                else:
                        out = self.mw.model(batch)
                        pred = out.argmax(dim=1)
                        correct += (pred == batch.y).sum().item()
                        if test:
                            running_loss += loss_fn(out, batch.y).item()
                            one_hot_labels = F.one_hot(batch.y, num_classes=out.shape[1])
                            # Apply softmax to get probabilities
                            out_mse = F.softmax(out, dim=1)
                            running_mse_loss += self.mw.mse_loss(out_mse, one_hot_labels.float()).item()
            
            loss = running_loss / len(loader)
            loss_mse = running_mse_loss / len(loader)
            accuracy = correct / len(loader.dataset)

        return accuracy, loss, loss_mse
    
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
            try:
                hidden_representations = self.mw.model.forward_2(self.data.x.to(self.device), self.data.edge_index.to(self.device), self.mw.ppr_matrix).detach().cpu().numpy()
            except:
                hidden_representations = self.mw.model(self.data.x.to(self.device), self.data.edge_index.to(self.device), self.mw.ppr_matrix).detach().cpu().numpy()
        
        return hidden_representations
    
    def save_training_results(self, results):
        """
        Enhanced result saving with progressive width information
        """
        file_exists = os.path.isfile(self.output_path)
        
        with open(self.output_path, 'a', newline='') as csvfile:
            fieldnames = ['model_type', 'layers', "hidden_channels", "epochs", 
                        'train_loss', 'train_mse_loss', 'train_accuracy', 
                        'test_loss', 'test_mse_loss', 'test_accuracy', 
                        'mad_value', 'mad_gap', 'dirichlet_energy', 'trial',
                        'learning_rate', 'progressive_width', 'model_complexity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            for result in results:
                writer.writerow(result)
    
    def save_training_results_madreg(self, results):
        """
        Enhanced result saving for MADReg experiments
        """
        file_exists = os.path.isfile(self.output_path)
        
        with open(self.output_path, 'a', newline='') as csvfile:
            fieldnames = ['model_type', 'layers', "hidden_channels", "epochs", 
                        'train_loss', 'total_loss', 'madreg_loss', 'train_mse_loss', 'train_accuracy', 
                        'test_loss', 'test_mse_loss', 'test_accuracy', 
                        'mad_value', 'mad_gap', 'dirichlet_energy', 'trial',
                        'learning_rate', 'progressive_width', 'model_complexity', 'lambda_reg']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            for result in results:
                writer.writerow(result)