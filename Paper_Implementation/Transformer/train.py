"""
Transformer 학습 코드 + 합성 데이터 데모

실제 번역 데이터셋(WMT, IWSLT 등) 없이도 모델 학습이 동작하는지
확인할 수 있도록 합성 시퀀스 reverse task로 데모 학습 수행.

Task: 입력 시퀀스를 뒤집어 출력하는 task
예) [1, 2, 3, 4] -> [4, 3, 2, 1]
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import Transformer
from utils import get_device, ResultLogger


PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


class ReverseDataset(Dataset):
    """시퀀스 뒤집기 task용 합성 데이터"""
    def __init__(self, num_samples=2000, vocab_size=20, seq_len=8):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        torch.manual_seed(42)
        self.data = []
        for _ in range(num_samples):
            src = torch.randint(3, vocab_size, (seq_len,))  # 0,1,2는 special tokens
            tgt = torch.cat([torch.tensor([BOS_IDX]), src.flip(0), torch.tensor([EOS_IDX])])
            self.data.append((src, tgt))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


def make_pad_mask(seq, pad_idx=PAD_IDX):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_subsequent_mask(seq):
    seq_len = seq.size(1)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask.unsqueeze(0).unsqueeze(0)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_pad_mask(src).to(device)
        tgt_mask = (make_pad_mask(tgt_in) & make_subsequent_mask(tgt_in).to(device))

        pred = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(pred.reshape(-1, pred.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(-1) == tgt_out).sum().item()
        total += tgt_out.numel()

    return total_loss / len(loader), correct / total


def main():
    device = get_device()
    print(f"Device: {device}")

    config = {
        "model": "Transformer (small)",
        "task": "Sequence Reverse (synthetic)",
        "vocab_size": 20,
        "d_model": 64,
        "heads": 4,
        "layers": 2,
        "d_ff": 128,
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 0.0005,
        "device": str(device),
    }

    train_dataset = ReverseDataset(num_samples=2000, vocab_size=20, seq_len=8)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    model = Transformer(
        src_vocab_size=20,
        tgt_vocab_size=20,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    logger = ResultLogger(results_dir="results", model_name="Transformer")

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.log(epoch=epoch, train_loss=train_loss, train_acc=train_acc)
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

    logger.save_all(model=model, config=config)
    print("\n실제 번역 task는 IWSLT/WMT 데이터셋 + 토크나이저 필요.")


if __name__ == "__main__":
    main()
