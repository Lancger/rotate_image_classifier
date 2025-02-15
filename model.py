import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class RotateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, 360)  # 360度分类

    def forward(self, x):
        """
        前向传播，使用resnet 50作为backbone，后面接一个线性层。

        Args:
            x: (batch, 3, 224, 224)

        Returns:
            tensor: 360维的一个tensor，表征属于每一个角度类别的概率
        """
        x = self.model(x)
        return x