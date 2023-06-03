import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights, ResNet50_Weights


def ResNet50(num_classes=10):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, num_classes)
    return model


def DenseNet121(num_classes=10):
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(1024, num_classes)
    return model
