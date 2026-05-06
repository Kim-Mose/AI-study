"""
BERT 학습 데모

작은 합성 데이터로 MLM(Masked Language Model) 사전학습 데모를 수행한다.
실제 사전학습은 위키피디아/BookCorpus 같은 대용량 데이터 + 며칠 학습 필요.

실무에서는 huggingface transformers의 사전학습된 BERT를 가져와 fine-tuning하는 게 일반적.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import BERT, BERTForMLM
from utils import get_device, ResultLogger


VOCAB_SIZE = 100
PAD_IDX = 0
MASK_IDX = 1
SPECIAL_TOKENS = 2  # 0=PAD, 1=MASK


class MLMDataset(Dataset):
    """MLM용 합성 데이터: 무작위 토큰 시퀀스 + 15% 마스킹"""
    def __init__(self, num_samples=1000, seq_len=20, vocab_size=VOCAB_SIZE):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        torch.manual_seed(42)
        self.data = []
        for _ in range(num_samples):
            tokens = torch.randint(SPECIAL_TOKENS, vocab_size, (seq_len,))
            self.data.append(tokens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = self.data[idx].clone()
        labels = torch.full_like(tokens, -100)  # -100은 loss 계산 시 무시

        # 15% 마스킹
        mask_prob = torch.rand(tokens.size())
        mask = mask_prob < 0.15
        labels[mask] = tokens[mask]
        tokens[mask] = MASK_IDX

        segment = torch.zeros_like(tokens)
        return tokens, segment, labels


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for tokens, segment, labels in loader:
        tokens, segment, labels = tokens.to(device), segment.to(device), labels.to(device)

        pred = model(tokens, segment)
        loss = criterion(pred.reshape(-1, pred.size(-1)), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # 마스킹된 위치만 정확도 계산
        mask_positions = labels != -100
        correct += (pred.argmax(-1)[mask_positions] == labels[mask_positions]).sum().item()
        total += mask_positions.sum().item()

    return total_loss / len(loader), correct / total if total > 0 else 0


def main():
    device = get_device()
    print(f"Device: {device}")

    config = {
        "model": "BERT (mini)",
        "task": "MLM 사전학습 (synthetic)",
        "vocab_size": VOCAB_SIZE,
        "d_model": 64,
        "heads": 4,
        "layers": 2,
        "d_ff": 128,
        "max_len": 20,
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 0.001,
        "device": str(device),
    }

    train_dataset = MLMDataset(num_samples=1000, seq_len=20, vocab_size=VOCAB_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    bert = BERT(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_len=20,
    ).to(device)

    mlm_model = BERTForMLM(bert, VOCAB_SIZE, d_model=64).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(mlm_model.parameters(), lr=config["learning_rate"])

    logger = ResultLogger(results_dir="results", model_name="BERT_mini")

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_epoch(mlm_model, train_loader, criterion, optimizer, device)
        logger.log(epoch=epoch, train_loss=train_loss, train_acc=train_acc)
        print(f"Epoch {epoch:2d} | MLM Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

    logger.save_all(model=mlm_model, config=config)
    print("\n실제 사전학습은 큰 코퍼스 + 며칠 학습 필요.")
    print("실무에서는 huggingface transformers 권장.")


if __name__ == "__main__":
    main()
