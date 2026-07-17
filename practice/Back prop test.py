import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Reproducibility
torch.manual_seed(0)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH = 128
EPOCHS = 3
HIDDEN = 128
LR = 0.02

# MNIST dataset
train_data = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH,
    shuffle=True
)

# Network parameters
W_x = (
    torch.randn(HIDDEN, 784, device=DEVICE) / HIDDEN**0.5
).requires_grad_()

b_h = torch.zeros(HIDDEN, device=DEVICE, requires_grad=True)

W_y = (
    torch.randn(10, HIDDEN, device=DEVICE) / 10**0.5
).requires_grad_()

b_y = torch.zeros(10, device=DEVICE, requires_grad=True)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):

    for images, labels in train_loader:

        x = images.view(images.size(0), -1).to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        h = F.relu(x @ W_x.t() + b_h)

        logits = h @ W_y.t() + b_y

        loss = F.cross_entropy(logits, labels)

        # Compute gradients
        loss.backward()

        # Gradient descent update
        with torch.no_grad():

            W_x -= LR * W_x.grad
            b_h -= LR * b_h.grad

            W_y -= LR * W_y.grad
            b_y -= LR * b_y.grad

            # Clear gradients
            W_x.grad.zero_()
            b_h.grad.zero_()
            W_y.grad.zero_()
            b_y.grad.zero_()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

test_data = datasets.MNIST(
    "./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH,
    shuffle=False
)

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        x = images.view(images.size(0), -1).to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        h = F.relu(x @ W_x.t() + b_h)
        logits = h @ W_y.t() + b_y

        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")