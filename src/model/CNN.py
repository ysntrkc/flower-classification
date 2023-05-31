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
        )

        # classifier layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 32 * 32)
        x = self.classifier(x)
        return x
