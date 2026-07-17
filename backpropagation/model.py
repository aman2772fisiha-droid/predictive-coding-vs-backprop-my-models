# This model is the code for one feed foreward neural network for MNIST classification
import torch
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 392),
            nn.ReLU(),

            nn.Linear(392, 196),
            nn.ReLU(),

            nn.Linear(196, 98),
            nn.ReLU(),

            nn.Linear(98, 49),
            nn.ReLU(),

            nn.Linear(49,10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x