from utils.graph_data import GraphDataset
from utils.oversmooth import OversmoothMeasure
from trainer import Trainer

import numpy as np
import os
import json
import argparse

import torch
import torch.nn.functional as F
import torch_geometric


current_folder = os.getcwd()
config_default = os.path.join(current_folder, 'configs', 'config_test.json') # default config file

def parse_args():
    parser = argparse.ArgumentParser(description="Model Wrapper for GNN with Hyperparameter Search")
    parser.add_argument('--config', type=str, default=config_default, help='Path to the configuration file')
    parser.add_argument('--trial', type=int, default=1, help='Trial number for parallel training (default: 1)')
    return parser.parse_args()

def load_hyperparameters(config_file):
    print(f"Loading hyperparameters from {config_file}")
    with open(config_file, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams

def construct_output_path(base_output_path, trial_num):
    """
    Construct the full output path with trial number and .csv extension
    
    Args:
        base_output_path (str): Base path ending with "_" (e.g., "test_training_log_1_")
        trial_num (int): Trial number
    
    Returns:
        str: Full path with trial number and .csv extension
    """
    if not base_output_path.endswith("_"):
        raise ValueError(f"Output path must end with '_', got: {base_output_path}")
    
    return f"{base_output_path}{trial_num}.csv"

def adjust_random_seed(base_seed, trial_num):
    """
    Adjust random seed based on trial number to ensure different initializations
    
    Args:
        base_seed (int): Base random seed from config
        trial_num (int): Trial number
    
    Returns:
        int: Adjusted random seed
    """
    return base_seed + trial_num - 1  # trial 1 -> base_seed, trial 2 -> base_seed + 1, etc.


if __name__ == '__main__':
    
    checkpoint_dir = os.path.join(current_folder, 'checkpoints')
    
    try:
        args = parse_args()
        config_file = args.config
        trial_num = args.trial
    except KeyError:
        current_folder = os.path.basename(os.getcwd())
        config_file = os.path.join(current_folder, 'configs', 'config_test.json')
        trial_num = 1
        print('Using default config file:', config_file)
        print('Using default trial number:', trial_num)
    except SystemError:
        print('Error: No config file provided')
        SystemExit(1)
    
    hyperparams = load_hyperparameters(config_file)
    
    # Get base output path from config
    base_output_path = hyperparams.get("output_path", "training_log_")
    
    # Validate and construct full output path
    try:
        full_output_path = construct_output_path(base_output_path, trial_num)
        print(f"Output will be saved to: {full_output_path}")
    except ValueError as e:
        print(f"Error in output path configuration: {e}")
        print("Please ensure output_path in config ends with '_'")
        SystemExit(1)
    
    # Adjust random seed for this trial
    base_seed = hyperparams.get("random_seed", 42)
    adjusted_seed = adjust_random_seed(base_seed, trial_num)
    
    print(f"Trial {trial_num}: Using random seed {adjusted_seed} (base: {base_seed})")
    
    # Set random seeds
    seed_np = np.random.seed(adjusted_seed) 
    seed_torch = torch.manual_seed(adjusted_seed)
    
    # Initialize the GraphDataset
    gd = GraphDataset(dataname=hyperparams.get("dataset", "cora"), 
                      raw_data_path=hyperparams.get("raw_data_path", None),
                      batch_size=hyperparams.get("batch_size", 32), 
                      shuffle=hyperparams.get("data_shuffle", True), 
                      noise_level=hyperparams.get("noise_level", 0.0),
                      test_size=hyperparams.get("test_size", 0.2),
                      val_size=hyperparams.get("val_size", 0.2),
                      random_state=adjusted_seed,  # Use adjusted seed here too
                      num_train_per_class = hyperparams.get("num_train_per_class", 20),)
    
    # Initialize the oversmooth measure
    osm = OversmoothMeasure()
    
    (data, num_features, out_channels, evaluator) = gd.get_dataset()
    
    # Create trial-specific checkpoint directory
    trial_checkpoint_dir = os.path.join(checkpoint_dir, f"trial_{trial_num}")
    os.makedirs(trial_checkpoint_dir, exist_ok=True)
    
    # Construct full output file path
    output_file = os.path.join(trial_checkpoint_dir, full_output_path)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Dataset Info:\n{data}")
    print(f"Train mask: {data.train_mask.sum()}")
    print(f"Trial: {trial_num}")
    print(f"Random seed: {adjusted_seed}")
    print(f"Output file: {output_file}")
    
    model_name = hyperparams.get("model_type", "GCN")
    
    # Check if the model is a GCNGPP model or not (important for the Trainer)
    gpp = True if model_name=="GCNGPP" or model_name=="GSGPP" or model_name=="GATGPP" else False 
    skip = True if hyperparams.get("skip")=="True" else False
    ppr = True if hyperparams.get("ppr")=="True" else False
    
    # Initialize the Trainer with trial-specific output path
    trainer = Trainer(data=data,
                        model_name=model_name,
                        num_features=num_features,
                        out_channels=out_channels, 
                        device=device,
                        output_path=output_file,  # Use trial-specific path
                        hyperparams=hyperparams,
                        evaluator=evaluator,
                        gpp=gpp,
                        skip=skip,
                        ppr=ppr,
                        trial_num=trial_num  # Pass trial number to trainer
                        )
    
    # Train the model
    trainer.hyperparameter_train()