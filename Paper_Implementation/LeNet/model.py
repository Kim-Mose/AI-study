import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 28x28x1 -> 28x28x6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 28x28x6 -> 14x14x6
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 14x14x6 -> 10x10x16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 10x10x16 -> 5x5x16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = LeNet()
    print(model)

    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)
