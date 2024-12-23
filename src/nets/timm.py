import torch
import torch.nn as nn
from timm.models import create_model


class Timm(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool):
        super().__init__()

        model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        self.backbone = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
