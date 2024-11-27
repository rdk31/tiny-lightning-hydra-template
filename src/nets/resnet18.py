import torch.nn as nn
from torchvision.models import resnet18


class Resnet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.base_model = resnet18(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.base_model(x)
