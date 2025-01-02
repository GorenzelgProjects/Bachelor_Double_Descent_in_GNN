import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import numpy as np

class NoisyCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, noise_rate=0.0, seed=42):
        """
        Custom CIFAR-10 dataset with adjustable label noise.

        :param root: Root directory of the dataset.
        :param train: Whether to load the training set or test set.
        :param transform: Transformations to apply to images.
        :param target_transform: Transformations to apply to labels.
        :param noise_rate: Proportion of labels to be randomly changed. 
                           E.g., noise_rate=0.2 means 20% of labels will be randomly altered.
        :param seed: Random seed for reproducibility.
        """
        self.cifar10 = CIFAR10(root=root, train=train, download=True, transform=transform, target_transform=target_transform)
        self.noise_rate = noise_rate
        self.seed = seed
        self._add_label_noise()

    def _add_label_noise(self):
        """
        Introduces label noise by randomly changing a specified percentage of labels.
        """
        np.random.seed(self.seed)
        targets = np.array(self.cifar10.targets)

        # Determine number of labels to change
        num_noisy_labels = int(self.noise_rate * len(targets))

        # Randomly select indices to corrupt
        noisy_indices = np.random.choice(len(targets), num_noisy_labels, replace=False)

        # Randomly assign new labels at the chosen indices
        for i in noisy_indices:
            original_label = targets[i]
            possible_labels = list(range(10))  # CIFAR-10 has 10 classes
            possible_labels.remove(original_label)
            new_label = np.random.choice(possible_labels)
            targets[i] = new_label

        self.cifar10.targets = targets.tolist()

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        return self.cifar10[idx]

# Set parameters for dataset
noise_rate = 0.2  # 20% of labels will be noisy
batch_size = 64

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create noisy CIFAR-10 dataset
noisy_train_dataset = NoisyCIFAR10(root='./data', train=True, transform=transform, noise_rate=noise_rate)
noisy_test_dataset = NoisyCIFAR10(root='./data', train=False, transform=transform, noise_rate=noise_rate)

# Create data loaders
train_loader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(noisy_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Check an example
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"Images batch shape: {images.shape}")
print(f"Labels batch shape: {labels.shape}")
print(f"Labels: {labels}")
