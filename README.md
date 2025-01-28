
# Research Code for [Double Descent in the context of GNN]

Welcome to the repository for **[Double Descent in the context of GNN]**! This project is a research-based implementation of [brief description of research objective]. The code is designed to ensure replicability of experiments and facilitate further exploration.

## Table of Contents

- [About](#about)
- [Setup and Requirements](#setup-and-requirements)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## About

This repository contains the code and resources for replicating the experiments conducted in our bachelor project at DTU Compute. The primary objectives of this project are:

- Replicate Double Descent in supervised deep learning tasks.
- Analyze how GNN-domain specific issues, such as oversmoothing, modulates the interpolation threshold in GNNs.
- In the context of Node Classification-  and Graph Property Prediction tasks, how does the Double Descent phenomenon manifest, and how robust is its appearance when influenced by graph-level characteristics and domain-specific factors?

## Setup and Requirements

### Prerequisites

Ensure you have the following installed:

- Python >= 3.8
- [List any other specific dependencies, e.g., PyTorch, DGL, etc.]

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/GorenzelgProjects/Bachelor_Double_Descent_in_GNN.git
   cd Bachelor_Double_Descent_in_GNN
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

To run the experiments:

1. Prepare the dataset:

   - Download the dataset from [provide source or dataset details].
   - Place the dataset in the `data/` directory.

2. Execute the main script:

   ```bash
   python main.py --config configs/experiment_config.yaml
   ```

3. Results will be saved in the `results/` directory.

### Configurations

Experiment configurations can be modified in the `configs/` directory. Each configuration file corresponds to a specific experiment setup.

## Documentation

Comprehensive documentation is available in the `docs/` directory and includes:

- Explanation of the code structure.
- Details on each module.
- Examples for customization.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

If you find this repository helpful for your research, please consider citing our work:

```
@article{yourcitation,
  title={Your Title},
  author={Your Name},
  journal={Journal/Conference Name},
  year={Year}
}
```

