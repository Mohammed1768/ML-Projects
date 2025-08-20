import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self):
        super.__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 10)
        )

    def feed_forward(self, x):
        self.flatten(x)
        return self.net(x)


