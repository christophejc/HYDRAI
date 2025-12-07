import torch
import torch.nn as nn

class HydrationClassifier(nn.Module):
    """
    Simple MLP classifier used for hydration prediction
    from NormWear embeddings.

    Args:
        input_dim (int): dimension of embedding input (e.g. 2304)
        num_classes (int): number of output hydration classes (3)
    """
    def __init__(self, input_dim=2304, num_classes=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)
