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
    batch_losses = []  # Loss for every mini-batch (Task 6)

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
            batch_losses.append(loss.item())

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

    return losses, accuracies, batch_losses


# Task 6: Training the Network

print("\n--- Task 6: Training the Network ---")

# Instantiate a new neural network
trained_model = SimpleNeuralNetwork()

# Train the network
training_losses, training_accuracies, batch_losses = train_network(
    trained_model,
    train_loader,
    epochs=5,
    learning_rate=0.01,
    momentum=0.9
)

# Print proof that the loss of every mini-batch was collected
print(f"Mini-batch loss values collected: {len(batch_losses)}")

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

# Extra graph showing all mini-batch losses collected during training
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(batch_losses) + 1), batch_losses, linewidth=0.7)
plt.title("Training Loss per Mini-batch")
plt.xlabel("Mini-batch Number")
plt.ylabel("Loss")
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

# Test different learning rates and momentum values with the same model/epochs.
parameter_settings = [
    (0.01, 0.9),
    (0.001, 0.9),
    (0.01, 0.5)
]

tuning_results = []
evaluation_criterion = nn.CrossEntropyLoss(reduction="sum")

for lr, mom in parameter_settings:
    print(f"\nTesting Learning Rate = {lr}, Momentum = {mom}")

    # New model for each experiment. Use the same seed for reproducibility.
    torch.manual_seed(42)
    tuning_model = SimpleNeuralNetwork()
    tuning_train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    losses, accuracies, _ = train_network(
        tuning_model,
        tuning_train_loader,
        epochs=3,
        learning_rate=lr,
        momentum=mom
    )

    # Evaluate accuracy and loss on the same MNIST test set.
    tuning_model.eval()
    correct = 0
    total = 0
    total_test_loss = 0.0
    with torch.no_grad():
        for images_test, labels_test in test_loader:
            outputs = tuning_model(images_test)
            total_test_loss += evaluation_criterion(outputs, labels_test).item()
            predicted = outputs.argmax(dim=1)
            total += labels_test.size(0)
            correct += (predicted == labels_test).sum().item()

    test_accuracy = 100 * correct / total
    test_loss = total_test_loss / total
    final_training_loss = losses[-1]
    tuning_results.append((lr, mom, final_training_loss, test_loss, test_accuracy))
    print(
        f"Result -> LR={lr}, Momentum={mom}, "
        f"Training Loss={final_training_loss:.4f}, "
        f"Test Loss={test_loss:.4f}, Test Accuracy={test_accuracy:.2f}%"
    )

print("\n--- Parameter Tuning Summary ---")
for lr, mom, train_loss, test_loss, accuracy in tuning_results:
    print(
        f"LR={lr}, Momentum={mom} | "
        f"Training Loss={train_loss:.4f} | "
        f"Test Loss={test_loss:.4f} | "
        f"Test Accuracy={accuracy:.2f}%"
    )

# Select by highest test accuracy, using lowest test loss to break ties.
best_parameters = max(tuning_results, key=lambda r: (r[4], -r[3]))
lowest_loss_parameters = min(tuning_results, key=lambda r: r[3])
best_lr, best_mom = best_parameters[0], best_parameters[1]
print(
    f"Best test accuracy: LR={best_lr}, Momentum={best_mom}, "
    f"Accuracy={best_parameters[4]:.2f}%"
)
print(
    f"Lowest test loss: LR={lowest_loss_parameters[0]}, "
    f"Momentum={lowest_loss_parameters[1]}, "
    f"Loss={lowest_loss_parameters[3]:.4f}"
)


# Task 8 - Part 2: Model Architecture Tuning
print("\n--- Task 8: Model Architecture Tuning ---")

# One hidden layer of 256 neurons: change neuron count only.
class WiderNeuralNetwork(nn.Module):
    def __init__(self):
        super(WiderNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Two hidden layers of 256 and 128 neurons: change layer count.
class TunedNeuralNetwork(nn.Module):
    def __init__(self):
        super(TunedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Compare three architectures, each with 3 epochs and identical settings.
# An independently seeded DataLoader gives each architecture the same image order.
architecture_settings = [
    ("Original 784 -> 128 -> 10", SimpleNeuralNetwork),
    ("Wider    784 -> 256 -> 10", WiderNeuralNetwork),
    ("Deeper   784 -> 256 -> 128 -> 10", TunedNeuralNetwork)
]
architecture_results = []

for architecture_name, network_class in architecture_settings:
    torch.manual_seed(42)
    architecture_model = network_class()
    architecture_train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nArchitecture: {architecture_name}")
    print(architecture_model)
    arch_losses, arch_accuracies, _ = train_network(
        architecture_model,
        architecture_train_loader,
        epochs=3,
        learning_rate=best_lr,
        momentum=best_mom
    )

    architecture_model.eval()
    correct = 0
    total = 0
    total_test_loss = 0.0
    with torch.no_grad():
        for images_test, labels_test in test_loader:
            outputs = architecture_model(images_test)
            total_test_loss += evaluation_criterion(outputs, labels_test).item()
            predicted = outputs.argmax(dim=1)
            total += labels_test.size(0)
            correct += (predicted == labels_test).sum().item()

    test_accuracy = 100 * correct / total
    test_loss = total_test_loss / total
    final_training_loss = arch_losses[-1]
    architecture_results.append(
        (architecture_name, final_training_loss, test_loss, test_accuracy)
    )

print("\n--- Architecture Tuning Summary ---")
print(f"All models: epochs=3, LR={best_lr}, Momentum={best_mom}")
for name, train_loss, test_loss, accuracy in architecture_results:
    print(
        f"{name} | Training Loss={train_loss:.4f} | "
        f"Test Loss={test_loss:.4f} | Test Accuracy={accuracy:.2f}%"
    )

best_architecture = max(architecture_results, key=lambda r: (r[3], -r[2]))
lowest_loss_architecture = min(architecture_results, key=lambda r: r[2])
print(f"Best test accuracy architecture: {best_architecture[0]} ({best_architecture[3]:.2f}%)")
print(f"Lowest test loss architecture: {lowest_loss_architecture[0]} ({lowest_loss_architecture[2]:.4f})")
