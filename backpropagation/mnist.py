""" 
train.py

Part 1
- Load MNIST (loads the mist daaset from pytorch)
- Create DataLoaders (they are functions that load images and labels more efficiently)
- Inspect the dataset(using the matplotlib.pyplot library to check if the image matches the label)
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import MNISTClassifier
from pathlib import Path
# -------------------------------------------------------
# Configuration (lets choose a batch size of 128)
# -------------------------------------------------------

BATCH_SIZE = 128

# -------------------------------------------------------
# Image Transform (transforms the image which is 28*28 array of pixel value numbers into a pytorch tensor)
# -------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
])

# -------------------------------------------------------
# Load Training Dataset (loads a dataset of 60,000 MNIST images from the pytorch library used to train our model)
# -------------------------------------------------------

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

# -------------------------------------------------------
# Load Test Dataset (loads a dataset of 10,000 MNIST imahes from the pytorch library used to test our model after training)
# -------------------------------------------------------

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

# -------------------------------------------------------
# DataLoaders (we set up two dataloaders, one for training and one for testing with the batch size we specified earlier and tell the loader the shuffle the order of the images for training set but not the testing set.)
# -------------------------------------------------------

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# -------------------------------------------------------
# Dataset Information
# -------------------------------------------------------

print("=" * 50)
print("Dataset Information")
print("=" * 50)

print(f"Training Images : {len(train_dataset)}")
print(f"Testing Images  : {len(test_dataset)}")

print()

# -------------------------------------------------------
# First Sample ( we take 1 sample image from the training set and check its shape)
# -------------------------------------------------------

image, label = train_dataset[0]

print(f"Image Shape : {image.shape}")
print(f"Label       : {label}")

print()

# -------------------------------------------------------
# First Batch (we take the first batch from the train loader and check its size)
# -------------------------------------------------------

images, labels = next(iter(train_loader))

print(f"Batch Images Shape : {images.shape}")
print(f"Batch Labels Shape : {labels.shape}")

print()

# -------------------------------------------------------
# Visualize First Image (we use matplotlib to display the pixels in greyscale for the first image in the training set and check if the lable is the same number)
# -------------------------------------------------------

plt.imshow(images[0].squeeze(), cmap="gray")
plt.title(f"Label : {labels[0].item()}")
plt.axis("off")
plt.show()



# -------------------------------------------------------
# Device (tells python where to load our training model. it will be cuda if torch.cuda is available, if not it will be loaded to our CPU)
# -------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# -------------------------------------------------------
# Hyperparameters ( We have to predetermine a small learning rate, and how many cycles should the training images train our model, which is the number of epochs.)
# -------------------------------------------------------

LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# -------------------------------------------------------
# Model(we call our predefined MNIST classifier model to the device we selected above.)
# -------------------------------------------------------

model = MNISTClassifier().to(device)

# -------------------------------------------------------
# Loss Function ( CrossEntropyLoss measures how different the model's predicted class scores are from the true labels. During loss.backward(), PyTorch automatically computes the gradients of this loss with respect to all trainable parameters. The optimizer then uses these gradients to update the parameters and reduce the loss.
# -------------------------------------------------------

criterion = torch.nn.CrossEntropyLoss()

# -------------------------------------------------------
# Optimizer (this is the function that fasttracks the parameter updates during gradient descent)
# -------------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# -------------------------------------------------------
# Model Summary these calculations show the total number of parameters in our model, and the number of trainable parameters in our model
# -------------------------------------------------------

print(model)

print()

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Total Parameters     : {total_parameters}")
print(f"Trainable Parameters : {trainable_parameters}")

print()

# -------------------------------------------------------
# Test Forward Pass (here we do one foreward pass just for one batch of 128 MNIST images.)
# -------------------------------------------------------

images, labels = next(iter(train_loader))

images = images.to(device)

outputs = model(images)

print(f"Input Shape  : {images.shape}")
print(f"Output Shape : {outputs.shape}")

print()

# -------------------------------------------------------
# Predicted Classes (Here we see the predictions of our model after only 128 images have gone through one foreward pass. the predictions are very off . uncomment next section to see test predictions) then we calculate the loss using the crossentropy we defined above and get our final loss
# -------------------------------------------------------

'''predictions = torch.argmax(outputs, dim=1)

print("Predictions")
print(predictions)

print()

print("Ground Truth")
print(labels)

print()

# -------------------------------------------------------
# Test Loss 
# -------------------------------------------------------

loss = criterion(
    outputs,
    labels.to(device),
)

print(f"Initial Loss : {loss.item():.4f}")

'''
# ============================================================
# PART 3
# Training Loop
# ============================================================


# ------------------------------------------------------------
# Training Configuration (tis creates a folder called checkpoints which tracks loss and accuracy)
# ------------------------------------------------------------

SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

best_accuracy = 0.0

train_loss_history = []
train_accuracy_history = []

test_loss_history = []
test_accuracy_history = []

# ============================================================
# Training Function (here we define what training and testing one epoch means(one whole cycle))
# ============================================================

def train_one_epoch(model,
                    dataloader,
                    criterion,
                    optimizer,
                    device):

    model.train() #tells pytorch to change to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        # ----------------------------------------
        # Clear previous gradients
        # ----------------------------------------
        optimizer.zero_grad()

        # ----------------------------------------
        # Forward pass
        # ----------------------------------------
        outputs = model(images)

        # ----------------------------------------
        # Compute loss
        # ----------------------------------------
        loss = criterion(outputs, labels)

        # ----------------------------------------
        # Backpropagation
        # ----------------------------------------
        loss.backward()

        # ----------------------------------------
        # Update parameters
        # ----------------------------------------
        optimizer.step()

        # ----------------------------------------
        # Statistics
        # ----------------------------------------
        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy

# ============================================================
# Evaluation Function
# ============================================================

@torch.no_grad()
def evaluate(model,
             dataloader,
             criterion,
             device):

    model.eval() # changes pytorch to change the model to predicting mode

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Main Training Loop
# ============================================================

print("=" * 60)
print("Starting Training")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_acc)

    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_acc)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc:.2f}% "
        f"| Test Loss: {test_loss:.4f} "
        f"| Test Acc: {test_acc:.2f}%"
    )

    if test_acc > best_accuracy:

        best_accuracy = test_acc

        torch.save(
            model.state_dict(),
            SAVE_DIR / "best_model.pth",
        )

        print(
            f"New best model saved "
            f"(Accuracy: {best_accuracy:.2f}%)"
        )

print()

print("=" * 60)
print("Training Finished")
print("=" * 60)

print(f"Best Test Accuracy : {best_accuracy:.2f}%")
print()

# ============================================================
# Load Best Model
# ============================================================

model.load_state_dict(
    torch.load(
        SAVE_DIR / "best_model.pth",
        map_location=device,
    )
)

print("Best model loaded successfully.")

# ============================================================
# Final Evaluation
# ============================================================

final_loss, final_accuracy = evaluate(
    model,
    test_loader,
    criterion,
    device,
)

print()
print("=" * 60)
print("Final Evaluation")
print("=" * 60)

print(f"Loss     : {final_loss:.4f}")
print(f"Accuracy : {final_accuracy:.2f}%")

print()

# ============================================================
# Show Some Predictions
# ============================================================

model.eval()

images, labels = next(iter(test_loader))

images = images.to(device)

outputs = model(images)

predictions = outputs.argmax(dim=1)

print("=" * 60)
print("Sample Predictions")
print("=" * 60)

for i in range(10):

    print(
        f"Image {i:2d} | "
        f"Prediction: {predictions[i].item()} | "
        f"Ground Truth: {labels[i].item()}"
    )  