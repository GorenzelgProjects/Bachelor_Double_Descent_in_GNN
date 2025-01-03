import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnetMNIST import make_resnet18k as ResNet18k  # Assuming resnet18k.py defines this model
import os
import csv

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation for MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load MNIST dataset
def load_dataset():
    dataset_path = './data/MNIST'
    #download = not (os.path.exists(dataset_path) and os.listdir(dataset_path))
    download = False
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=download, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=download, transform=transform)

    return trainset, testset

# Define training function
def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs, checkpoint_path, csv_file, k, dataset_fraction):
    start_epoch = 0
    
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
            writer.writerow(['Model Size', 'Epoch', 'Train Loss', 'Dataset Fraction', 'Train Accuracy', 'Test Accuracy', 'Test Loss'])
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
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

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}")
                running_loss = 0.0
        
        train_loss = running_loss / len(trainloader)
        train_accuracy = 100. * correct / total
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_loss /= len(testloader)
        test_accuracy = 100. * correct / total
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
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
            writer.writerow([k, epoch + 1, train_loss, dataset_fraction, train_accuracy, test_accuracy, test_loss])

# Main function
def main():
    trainset, testset = load_dataset()

    # Create directory for checkpoints
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # CSV file for logging
    csv_file = os.path.join(checkpoint_dir, 'training_log.csv')

    # Iterate over different values of k
    start_i, start_j = 0, 1
    checkpoint_found = False

    # Check for existing checkpoints to resume
    for i in range(32):
        k = 1 if i == 0 else 2 * i
        for j in range(1, 11):
            checkpoint_path = f'./mnist_checkpoint_k{k}_fraction{j}.pth'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                start_i, start_j = i, j
                checkpoint_found = True
                print(f"Resuming training from k={k}, dataset fraction={j/10.0}, epoch {checkpoint['epoch'] + 1}")
                break
        if checkpoint_found:
            break

    for i in range(start_i, 32):
        k = 1 if i == 0 else 2 * i
        print(f"Training model with k={k}")

        # Iterate over different dataset sizes
        for j in range(start_j if i == start_i else 1, 11):
            dataset_fraction = j / 10.0
            subset_size = int(len(trainset) * dataset_fraction)
            train_subset, _ = torch.utils.data.random_split(trainset, [subset_size, len(trainset) - subset_size])
            trainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

            print(f"Training with dataset fraction: {dataset_fraction}")

            # Initialize the model, criterion, and optimizer
            model = ResNet18k(k=k).to(device)  # Assuming the ResNet18k function accepts k as a parameter
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

            # Load checkpoint if available
            checkpoint_path = f'./mnist_checkpoint_k{k}_fraction{j}.pth'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch}")
            else:
                start_epoch = 0

            # Train the model
            train_model(model, trainloader, testloader, criterion, optimizer, num_epochs=10, checkpoint_path=checkpoint_path, csv_file=csv_file, k=k, dataset_fraction=dataset_fraction)

if __name__ == "__main__":
    main()