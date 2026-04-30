import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net (2015)
    의료 영상 세그멘테이션을 위한 모델.
    인코더(다운샘플링) + 디코더(업샘플링) + 스킵 연결 구조.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 인코더 (Down)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 가장 깊은 부분 (Bottleneck)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 디코더 (Up)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 인코더
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # 역순으로

        # 디코더
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # 업샘플링
            skip = skip_connections[idx // 2]

            # 크기 안 맞으면 보정
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            concat = torch.cat((skip, x), dim=1)  # 스킵 연결
            x = self.ups[idx + 1](concat)

        return self.final_conv(x)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    print(model(x).shape)
