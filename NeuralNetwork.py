from torch import nn


class NeuralNetwork(nn.Module):
    """
        Creates the LCD model architecture and implements the forward pass function.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 128), 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
