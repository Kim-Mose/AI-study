"""
U-Net 학습 코드 + 합성 데이터셋 데모

세그멘테이션 데이터셋(Carvana, Oxford Pets 등)이 없어도
모델이 동작하는지 확인할 수 있도록 합성 데이터로 데모 학습을 수행한다.

실제 학습은 자신의 데이터셋을 SegmentationDataset에 연결해야 한다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import UNet
from utils import get_device, ResultLogger


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def dice_score(pred, target, smooth=1.0):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class SyntheticSegDataset(Dataset):
    """합성 세그멘테이션 데이터: 원/사각형 모양을 마스크로"""
    def __init__(self, num_samples=200, size=64):
        self.num_samples = num_samples
        self.size = size
        torch.manual_seed(42)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        size = self.size
        img = torch.zeros(3, size, size)
        mask = torch.zeros(1, size, size)

        # 원 그리기
        cx, cy = torch.randint(size//4, 3*size//4, (2,)).tolist()
        radius = torch.randint(size//8, size//4, (1,)).item()
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        circle = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2

        img[0] = circle.float() * 0.8 + torch.randn(size, size) * 0.1
        img[1] = circle.float() * 0.5 + torch.randn(size, size) * 0.1
        img[2] = circle.float() * 0.3 + torch.randn(size, size) * 0.1
        mask[0] = circle.float()

        return img, mask


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_dice = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += dice_score(pred, y).item()
    return total_loss / len(loader), total_dice / len(loader)


def main():
    device = get_device()
    print(f"Device: {device}")

    config = {
        "model": "U-Net",
        "dataset": "Synthetic (circles)",
        "image_size": 64,
        "batch_size": 8,
        "epochs": 20,
        "learning_rate": 0.001,
        "loss": "BCEWithLogits",
        "device": str(device),
    }

    train_dataset = SyntheticSegDataset(num_samples=200, size=64)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    logger = ResultLogger(results_dir="results", model_name="UNet")

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_dice = train(model, train_loader, criterion, optimizer, device)
        logger.log(epoch=epoch, train_loss=train_loss, train_acc=train_dice)
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")

    logger.save_all(model=model, config=config)
    print("\n실제 데이터셋(Carvana 등)으로 학습하려면 SyntheticSegDataset을")
    print("SegmentationDataset으로 교체해주세요.")


if __name__ == "__main__":
    main()
