"""
U-Net 학습 코드
세그멘테이션 데이터셋이 필요하므로 여기서는 학습 루프 템플릿만 제공한다.
실제 사용 시에는 자신의 데이터셋(이미지, 마스크)을 만들어야 한다.

추천 데이터셋:
- Carvana (캐글): 자동차 배경 제거
- Oxford Pets: 동물 세그멘테이션
- ISIC: 피부 병변 세그멘테이션
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import UNet


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


class SegmentationDataset(Dataset):
    """
    세그멘테이션용 데이터셋 템플릿.
    이미지와 마스크 폴더를 받아서 처리한다.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 실제 사용 시 파일 목록을 채울 것
        self.images = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 이미지와 마스크 로드
        # 실제 구현은 데이터셋에 맞게 작성
        pass


def dice_loss(pred, target, smooth=1.0):
    """세그멘테이션에 자주 쓰이는 Dice Loss"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device).float()
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = get_device()
    print(f"Device: {device}")

    EPOCHS = 30
    LR = 0.0001

    # 데이터셋 로드 (실제 사용 시 작성)
    # train_dataset = SegmentationDataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # 또는 dice_loss
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 학습 루프
    # for epoch in range(1, EPOCHS + 1):
    #     loss = train(model, train_loader, criterion, optimizer, device)
    #     print(f"Epoch {epoch} | Loss: {loss:.4f}")

    print("U-Net 모델 정의 완료. 데이터셋 연결 필요.")


if __name__ == "__main__":
    main()
