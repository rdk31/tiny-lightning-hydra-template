import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class Resnet18(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool):
        super().__init__()

        if pretrained:
            self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            if num_classes != 1000:
                self.base_model.fc = nn.Linear(
                    self.base_model.fc.in_features, num_classes
                )
        else:
            self.base_model = resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.base_model(x)
