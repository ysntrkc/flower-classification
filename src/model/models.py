import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights, ResNet18_Weights, EfficientNet_B0_Weights


def ResNet18(num_classes=10):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    features = list(model.fc.parameters())[0].shape[1]
    model.fc = nn.Linear(features, num_classes)
    return model


def DenseNet121(num_classes=10):
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    features = list(model.classifier.parameters())[0].shape[1]
    model.classifier = nn.Linear(features, num_classes)
    return model


def EfficientNetB0(num_classes=10):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    features = list(model.classifier.parameters())[0].shape[1]
    model.classifier = nn.Linear(features, num_classes)
    return model
