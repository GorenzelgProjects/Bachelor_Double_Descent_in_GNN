import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet18k import make_resnet18k as ResNet18k   # Assuming resnet18k.py defines this model
from cifar_label_noise_distribution import NoisyCIFAR10
import os
import csv
import numpy as np

#set seed
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Device: {device}")

# Set parameters for dataset
noise_rate = 0.15  # 15% of labels will be noisy
#batch_size = 64

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Load MNIST dataset
def load_dataset(noise_rate=0.0, transform=transform):
    dataset_path = './data/CIFAR10_noisy'
    download = not (os.path.exists(dataset_path) and os.listdir(dataset_path))
    
    # Create noisy CIFAR-10 dataset
    trainset = NoisyCIFAR10(root='./data', train=True, transform=transform, noise_rate=noise_rate)
    testset = NoisyCIFAR10(root='./data', train=False, transform=transform, noise_rate=0)
    
    return trainset, testset

# Define training function
def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs, checkpoint_path, csv_file, k, dataset_fraction):
    start_epoch = 0
    

    # Initialize MSE loss criterion
    mse_loss_criterion = nn.MSELoss()
    
    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    # Open or create CSV file and write headers
    csv_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(['Model Size', 'Dataset Fraction', 'Epoch', 'Train Loss', 'Train Accuracy', 
                             'Test Accuracy', 'Test Loss', 'Train MSE Loss', 'Test MSE Loss'])
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        train_mse_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            
            with torch.no_grad():
                softmax_outputs = nn.functional.softmax(outputs, dim=1)  # Apply softmax to logits
                labels_one_hot = nn.functional.one_hot(labels, num_classes=outputs.size(1)).float()
                train_mse_loss += mse_loss_criterion(softmax_outputs, labels_one_hot).item()

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            #if i % 100 == 99:  # Print every 100 mini-batches
            #    print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}")
            #    running_loss = 0.0
        
        train_loss = running_loss / len(trainloader)
        train_mse_loss /= len(trainloader)
        train_accuracy = 100. * correct / total
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train MSE Loss: {train_mse_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        test_mse_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Compute MSE loss (not used for optimization)
                softmax_outputs = nn.functional.softmax(outputs, dim=1)  # Apply softmax to logits
                labels_one_hot = nn.functional.one_hot(labels, num_classes=outputs.size(1)).float()
                test_mse_loss += mse_loss_criterion(softmax_outputs, labels_one_hot).item()


                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(testloader)
        test_mse_loss /= len(testloader)
        test_accuracy = 100. * correct / total
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Test MSE Loss: {test_mse_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Save checkpoint at the end of each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

        # Log training progress to CSV file
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([k, dataset_fraction, epoch + 1, train_loss, train_accuracy, test_accuracy, test_loss, train_mse_loss, test_mse_loss])

# Main function
def main():
    # Create directory for checkpoints
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # CSV file for logging
    csv_file = os.path.join(checkpoint_dir, 'training_log_cifar_2.csv')

    # Iterate over different values of k
    start_i, start_j = 0, 1
    checkpoint_found = False
    use_checkpoint = False

    # Check for existing checkpoints to resume
    for i in range(start_i+26,44):
        k = 1 if i == 0 else i
        for j in range(1, 4):
            checkpoint_path = os.path.join(checkpoint_dir, f'cifar_checkpoint_k{k}_fraction{j}_experiment_2.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                start_i, start_j = i, j
                checkpoint_found = True
                print(f"Resuming training from k={k}, dataset fraction={j/3.0}, epoch {checkpoint['epoch'] + 1}")
                break
        if checkpoint_found:
            break

    for i in range(start_i+26, 44):
        k = 1 if i == 0 else i
        print(f"Training model with k={k}")

        # Iterate over different dataset sizes
        for j in range(start_j if i == start_i else 1, 4):
            dataset_fraction = j / 3.0
            trainset, testset = load_dataset(noise_rate=noise_rate)
            #trainset, testset = load_dataset(noise_rate=0.0)
            subset_size = int(len(trainset) * dataset_fraction)
            train_subset, _ = torch.utils.data.random_split(trainset, [subset_size, len(trainset) - subset_size])
            trainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

            print(f"Training with dataset fraction: {dataset_fraction}")

            # Initialize the model, criterion, and optimizer
            model = ResNet18k(k=k).to(device)  # Assuming the ResNet18k function accepts k as a parameter
            criterion = nn.CrossEntropyLoss()
            #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

            # Load checkpoint if available
            checkpoint_path = os.path.join(checkpoint_dir, f'cifar_checkpoint_k{k}_fraction{j}_experiment_2.pth')
            if os.path.exists(checkpoint_path) and use_checkpoint:
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch}")
            else:
                start_epoch = 0

            # Train the model
            train_model(model, trainloader, testloader, criterion, optimizer, num_epochs=1000, checkpoint_path=checkpoint_path, csv_file=csv_file, k=k, dataset_fraction=dataset_fraction)

if __name__ == "__main__":
    main()