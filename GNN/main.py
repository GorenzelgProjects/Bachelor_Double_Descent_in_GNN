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
    return parser.parse_args()

def load_hyperparameters(config_file):
    print(f"Loading hyperparameters from {config_file}")
    with open(config_file, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams


if __name__ == '__main__':
    
    #current_folder = os.path.basename(os.getcwd()) 
    checkpoint_dir = os.path.join(current_folder, 'checkpoints')
    
    try:
        config_file = parse_args().config
    except KeyError:
        #config_file = 'GNN_double_descent/config_1.json'
        current_folder = os.path.basename(os.getcwd())
        config_file = os.path.join(current_folder, 'configs', 'config_test.json')
        print('Using default config file:', config_file)
    except SystemError:
        print('Error: No config file provided')
        SystemExit(1)
    
    hyperparams = load_hyperparameters(config_file)
    
    seed_np = np.random.seed(hyperparams.get("random_seed", 42)) 
    seed_torch = torch.manual_seed(hyperparams.get("random_seed", 42))
    
    # Initialize the GraphDataset
    gd = GraphDataset(dataname=hyperparams.get("dataset", "cora"), 
                      raw_data_path=hyperparams.get("raw_data_path", None),
                      batch_size=hyperparams.get("batch_size", 32), 
                      shuffle=hyperparams.get("data_shuffle", True), 
                      noise_level=hyperparams.get("noise_level", 0.0),
                      test_size=hyperparams.get("test_size", 0.2),
                      val_size=hyperparams.get("val_size", 0.2),
                      random_state=hyperparams.get("random_seed", 42))
    
    # Initialize the oversmooth measure
    osm = OversmoothMeasure()
    
    (data, num_features, out_channels, evaluator) = gd.get_dataset()
    
    # get the output path from the hyperparameters
    output_path = hyperparams.get("output_path", "output.csv")
    output_file = os.path.join(checkpoint_dir, output_path)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    print(f"Dataset Info:\n{data}")
    
    
    model_name = hyperparams.get("model_type", "GCN")
    
    # Check if the model is a GCNGPP model or not (important for the Trainer)
    gpp = True if model_name=="GCNGPP" or model_name=="GSGPP" or model_name=="GATGPP" else False 
    skip = True if hyperparams.get("skip")=="True" else False
    ppr = True if hyperparams.get("ppr")=="True" else False
    
    
    # Initialize the Trainer
    trainer = Trainer(data=data,
                        model_name=model_name,
                        num_features=num_features,
                        out_channels=out_channels, 
                        device=device,
                        output_path=output_file,
                        hyperparams=hyperparams,
                        evaluator=evaluator,
                        gpp=gpp,
                        skip=skip,
                        ppr=ppr
                        )
    
    # Train the model
    trainer.hyperparameter_train()