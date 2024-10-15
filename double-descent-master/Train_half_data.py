import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet18k import make_resnet18k as ResNet18k  # Assuming resnet18k.py defines this model
import os
import csv

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation for CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

def load_data(train_size, test_size, batch_size_train=128, batch_size_test=100):

    #Calculate the size of the training and test data as % of the original data
    train_size = int(train_size * 50000)
    test_size = int(test_size * 10000)

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=False, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(range(0, train_size)))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(range(0, test_size)))

    return trainloader, testloader


trainloader, testloader = load_data(0.5, 0.5)

"""
# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

#Only use half of the data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(range(0, 25000)))
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(range(0, 5000)))

#Only use a quarter of the data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(range(0, 12500)))
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(range(0, 2500)))
"""
# Define loss function
criterion = nn.CrossEntropyLoss()

# Function to train and test the model while saving metrics to CSV
def train_and_test(k, num_epochs=1000, checkpoint_dir='./checkpoints', csv_file='training_metrics.csv'):
    # Create the model with varying k
    model = ResNet18k(k=k).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Create directory for checkpoints
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Open or create CSV file and write headers
    csv_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(['Model Size', 'Epoch', 'Train Loss', 'Test Loss', 'Train Error', 'Test Error'])

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_error = 100 * (1 - correct_train / total_train)

        # Test the model and get test loss and error
        test_loss, test_error = test_model(model, testloader)

        # Save checkpoint every 50 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(checkpoint_dir, f"resnet18k_k{k}_epoch{epoch}.pth"))

        # Log metrics to CSV file
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([k, epoch + 1, train_loss, test_loss, train_error, test_error])

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Error: {train_error}%, Test Error: {test_error}%")

def test_model(model, testloader):
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

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(testloader)
    test_error = 100 * (1 - correct / total)
    return test_loss, test_error
if __name__ == '__main__':
    # Assuming this is where you call the train_and_test function
    #for k in [2**i for i in range(6)]:  # k from 1, 2, 4, ..., 64
    #    print(f"Training ResNet18k with k={k}")
        #train_and_test(k, num_epochs=15)
    train_and_test(64, num_epochs=100)
