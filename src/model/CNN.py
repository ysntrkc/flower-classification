import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # convolutional feature extractor layer (128x128x3)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(16384, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # convolutional feature extractor
        x = self.features(x)

        # flatten
        x = x.view(x.size(0), -1)

        # classifier
        x = self.classifier(x)

        return x
