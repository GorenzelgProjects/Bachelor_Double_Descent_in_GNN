# Research Code for Double Descent in Graph Neural Network

Welcome to the repository for **Double Descent in Graph Neural Network (GNN)**! This project explores the Double Descent phenomenon in GNNs, analyzing how different factors influence generalization and model performance. The code is structured to facilitate experimentation and replicability.

## Table of Contents

- [About](#about)
- [Setup and Requirements](#setup-and-requirements)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## About

This repository is dedicated to studying **Double Descent in Graph Neural Networks**, with a focus on:

- Replicating the Double Descent phenomenon in supervised deep learning tasks for GNNs.
- Investigating how GNN-specific challenges (e.g., oversmoothing) affect the interpolation threshold.
- Evaluating the robustness of Double Descent in **Node Classification** and **Graph Property Prediction** tasks under different graph characteristics and domain-specific settings.

## Setup and Requirements

### Prerequisites

Ensure you have the following installed:

- Python >= 3.8
- The list of dependencies is found in `requirements.txt`.

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-repository/Bachelor_Double_Descent_in_GNN.git
   cd Bachelor_Double_Descent_in_GNN/GNN
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the experiments, execute the following command:

```bash
python main.py --config "GNN_double_descent/configs/config_test.json"
```

This will run the main experiment with the specified configuration file.

### Key Components

- **`main.py`**: The primary script to run experiments.
- **`trainer/`**: Contains training settings and loops for tasks such as node classification and graph property prediction.
- **`model_wrapper.py`**: A model builder class that constructs models based on specified configurations.
- **`models/`**: Includes the GNN models and related architectures.

## Code Structure

The code is organized as follows:

```
Bachelor_Double_Descent_in_GNN/
└── GNN/
    ├── checkpoints/            # Training checkpoints and logs  
    ├── configs/                # Configuration files for experiments  
    ├── data/                   # Datasets for training and evaluation  
    ├── experiments/            # Scripts for running experiments  
    ├── models/                 # Model definitions  
    │   ├── conventional_models.py  
    │   └── model.py  
    ├── trainer/                # Training scripts  
    ├── utils/                  # Utility functions and helpers  
    ├── visualizations/         # Scripts for visualizing results  
    ├── main.py                 # Main entry point for experiments  
    ├── model_wrapper.py        # Model construction and handling  
    └── trainer.py              # Training logic and settings  
```

## Documentation

Detailed documentation is provided in the `docs/` directory, covering:

- Code structure and module explanations.
- Customization examples for different models and tasks.
- How to extend the repository for new experiments.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make your changes and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

