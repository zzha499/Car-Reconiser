import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()

        # 1st Layer - Input data 224 * 224 * 3
        self.features = nn.Sequential(

            # 2nd Layer - Convolution layer 55 * 55 * 96 (max pooling)
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 3rd Layer - Convolution layer 27 * 27 * 256 (max pooling)
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 4th Layer - Convolution layer 13 * 13 * 384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 5th Layer - Convolution layer 13 * 13 * 384
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 6th Layer - Convolution layer 13 * 13 * 384 (max pooling)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),

            # 7th Layer - Fully connected layer
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # 8th Layer - Fully connected layer
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # 9th Layer - Fully connected layer with the number of classes as outputs
            nn.Linear(4096, num_classes),
        )

    # Forward pass function of the network
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Wrapper function for creating alexnet in model_initializer
def alexnet(**kwargs):
    return AlexNet(**kwargs)