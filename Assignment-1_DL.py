# ARTI 502 - Deep Learning
# Assignment 1: Training-Testing Neural Network with MNIST Dataset

import torch
import torchvision
import matplotlib

print("PyTorch loaded successfully!")
print("PyTorch version:", torch.__version__)


# Task 2: Download and Load the MNIST Dataset

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Convert MNIST images to PyTorch tensors
transform = transforms.ToTensor()

# Download and load the training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Download and load the testing dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("\nMNIST dataset loaded successfully!")
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))




# Task 3: Inspect the Dataset

# Get one batch of images and labels
images, labels = next(iter(train_loader))

print("\n--- Task 3: Inspect the Dataset ---")
print("Images tensor size:", images.shape)
print("Labels tensor size:", labels.shape)

# Visualize sample MNIST images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray")
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis("off")

plt.suptitle("Sample Images from MNIST Dataset")
plt.tight_layout()
plt.show()


# Task 4: Creating a Simple Neural Network

import torch.nn as nn
import torch.nn.functional as F

class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()

        # Input layer: 28 x 28 = 784 pixels
        # One hidden layer with 128 neurons
        self.fc1 = nn.Linear(784, 128)

        # Output layer: 10 classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten from [batch, 1, 28, 28] to [batch, 784]
        x = torch.flatten(x, 1)

        # Hidden layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Output layer
        x = self.fc2(x)

        return x

    # Create and verify the network
model = SimpleNeuralNetwork()

print("\n--- Task 4: Simple Neural Network ---")
print(model)

# Pass one batch through the network
sample_output = model(images)

print("\nInput images shape:", images.shape)
print("Output shape:", sample_output.shape)


 # Task 5: Training Function

import torch.optim as optim

def train_network(net, train_loader, epochs=5, learning_rate=0.01, momentum=0.9):

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # SGD optimizer with learning rate and momentum
    optimizer = optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    losses = []
    accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        net.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Calculate statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Print debug information every 500 batches
            if (batch_idx + 1) % 500 == 0:
                current_accuracy = 100 * correct / total

                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Current Loss: {loss.item():.4f}, "
                    f"Accuracy: {current_accuracy:.2f}%"
                )

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] Completed - "
            f"Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )

    return losses, accuracies


# Task 6: Training the Network

print("\n--- Task 6: Training the Network ---")

# Instantiate a new neural network
trained_model = SimpleNeuralNetwork()

# Train the network
training_losses, training_accuracies = train_network(
    trained_model,
    train_loader,
    epochs=5,
    learning_rate=0.01,
    momentum=0.9
)

# Plot training Loss and Accuracy

epochs_range = range(1, len(training_losses) + 1)

# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, training_losses, marker="o")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs_range)
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, training_accuracies, marker="o")
plt.title("Training Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(epochs_range)
plt.grid(True)
plt.tight_layout()
plt.show()


# Task 7: Testing the Network

print("\n--- Task 7: Testing the Network ---")

trained_model.eval()

correct = 0
total = 0

# Disable gradient calculation during testing
with torch.no_grad():
    for images_test, labels_test in test_loader:

        outputs = trained_model(images_test)

        _, predicted = torch.max(outputs, 1)

        total += labels_test.size(0)
        correct += (predicted == labels_test).sum().item()

# Calculate overall test accuracy
test_accuracy = 100 * correct / total

print(f"Correct predictions: {correct}/{total}")
print(f"Test Accuracy: {test_accuracy:.2f}%")



# Task 8: Tuning Parameters and Model Architecture

print("\n--- Task 8: Parameter Tuning ---")

# Different learning rate and momentum combinations
parameter_settings = [
    (0.01, 0.9),
    (0.001, 0.9),
    (0.01, 0.5)
]

tuning_results = []

for lr, mom in parameter_settings:

    print(f"\nTesting Learning Rate = {lr}, Momentum = {mom}")

    # Create a new model for each experiment
    tuning_model = SimpleNeuralNetwork()

    # Train the model
    losses, accuracies = train_network(
        tuning_model,
        train_loader,
        epochs=3,
        learning_rate=lr,
        momentum=mom
    )

    # Test the trained model
    tuning_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images_test, labels_test in test_loader:

            outputs = tuning_model(images_test)
            _, predicted = torch.max(outputs, 1)

            total += labels_test.size(0)
            correct += (predicted == labels_test).sum().item()

    accuracy = 100 * correct / total

    tuning_results.append((lr, mom, accuracy))

    print(
        f"Result -> Learning Rate: {lr}, "
        f"Momentum: {mom}, "
        f"Test Accuracy: {accuracy:.2f}%"
    )

print("\n--- Parameter Tuning Summary ---")

for lr, mom, accuracy in tuning_results:
    print(
        f"Learning Rate = {lr}, "
        f"Momentum = {mom} -> "
        f"Test Accuracy = {accuracy:.2f}%"
    )


    # Task 8 - Part 2: Model Architecture Tuning

print("\n--- Task 8: Model Architecture Tuning ---")

# Model with two hidden layers
class TunedNeuralNetwork(nn.Module):

    def __init__(self):
        super(TunedNeuralNetwork, self).__init__()

        # First hidden layer: 784 -> 256 nodes
        self.fc1 = nn.Linear(784, 256)

        # Second hidden layer: 256 -> 128 nodes
        self.fc2 = nn.Linear(256, 128)

        # Output layer: 128 -> 10 classes
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):

        # Flatten the image
        x = torch.flatten(x, 1)

        # First hidden layer
        x = F.relu(self.fc1(x))

        # Second hidden layer
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)

        return x


# Create the tuned model
tuned_architecture_model = TunedNeuralNetwork()

print("\nTuned Model Architecture:")
print(tuned_architecture_model)

# Train using the best parameters found above
architecture_losses, architecture_accuracies = train_network(
    tuned_architecture_model,
    train_loader,
    epochs=3,
    learning_rate=0.01,
    momentum=0.9
)

# Test the tuned architecture
tuned_architecture_model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images_test, labels_test in test_loader:

        outputs = tuned_architecture_model(images_test)
        _, predicted = torch.max(outputs, 1)

        total += labels_test.size(0)
        correct += (predicted == labels_test).sum().item()

architecture_test_accuracy = 100 * correct / total

print("\n--- Architecture Tuning Result ---")
print("Original Architecture: 784 -> 128 -> 10")
print("Tuned Architecture:    784 -> 256 -> 128 -> 10")
print(f"Tuned Architecture Test Accuracy: {architecture_test_accuracy:.2f}%")