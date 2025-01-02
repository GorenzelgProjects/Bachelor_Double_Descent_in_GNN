import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter

class NoisyCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, noise_rate=0.0, seed=42):
        """
        Custom CIFAR-10 dataset with adjustable label noise following the original label distribution.

        :param root: Root directory of the dataset.
        :param train: Whether to load the training set or test set.
        :param transform: Transformations to apply to images.
        :param target_transform: Transformations to apply to labels.
        :param noise_rate: Proportion of labels to be randomly changed.
        :param seed: Random seed for reproducibility.
        """
        self.cifar10 = CIFAR10(root=root, train=train, download=True, transform=transform, target_transform=target_transform)
        self.noise_rate = noise_rate
        self.seed = seed
        self._add_label_noise()

    def _add_label_noise(self):
        """
        Introduces label noise by randomly changing a specified percentage of labels,
        following the original label distribution. Also calculates and prints the effective noise rate.
        """
        np.random.seed(self.seed)
        targets = np.array(self.cifar10.targets)
        original_distribution = Counter(targets)

        # Determine number of labels to change based on noise rate
        num_noisy_labels = int(self.noise_rate * len(targets))
        altered_labels_count = Counter()  # Keep track of altered labels by class

        # Compute the number of noisy labels to assign per class, based on the original distribution
        noisy_labels_per_class = {
            label: int(count / len(targets) * num_noisy_labels)
            for label, count in original_distribution.items()
        }

        same_label_replacements = 0  # Track the number of replacements with the same label

        for label, count in noisy_labels_per_class.items():
            # Get indices of the current label to change
            label_indices = np.where(targets == label)[0]
            noisy_indices = np.random.choice(label_indices, count, replace=False)

            for idx in noisy_indices:
                original_label = targets[idx]
                possible_labels = list(range(10))  # CIFAR-10 has 10 classes
                possible_labels.remove(original_label)
                new_label = np.random.choice(possible_labels)

                if new_label == original_label:
                    same_label_replacements += 1
                else:
                    targets[idx] = new_label
                    altered_labels_count[original_label] += 1

        self.cifar10.targets = targets.tolist()

        # Calculate and print the effective noise rate
        actual_noisy_labels = num_noisy_labels - same_label_replacements
        effective_noise_rate = actual_noisy_labels / len(self.cifar10.targets)
        print(f"Requested noise rate: {self.noise_rate * 100}%")
        print(f"Effective noise rate after same-label replacements: {effective_noise_rate * 100:.2f}%")
        print(f"Same-label replacements: {same_label_replacements}")

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        return self.cifar10[idx]

if __name__ == "__main__":
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
