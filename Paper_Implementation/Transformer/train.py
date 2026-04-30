"""
Transformer 학습 코드 템플릿
실제로는 번역 데이터셋(예: WMT, IWSLT)이 필요하다.

여기서는 학습 루프 구조만 보여준다.
실제 사용 시에는 데이터 로더와 토크나이저를 연결해야 한다.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from model import Transformer


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def make_pad_mask(seq, pad_idx=0):
    """패딩 마스크 생성"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_subsequent_mask(seq):
    """디코더에서 미래 토큰 마스킹"""
    seq_len = seq.size(1)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask.unsqueeze(0).unsqueeze(0)


def train_epoch(model, loader, criterion, optimizer, device, pad_idx=0):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        # 디코더 입력은 마지막 토큰 제외, 정답은 첫 토큰 제외
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_pad_mask(src, pad_idx).to(device)
        tgt_mask = (make_pad_mask(tgt_in, pad_idx) & make_subsequent_mask(tgt_in).to(device))

        pred = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(pred.reshape(-1, pred.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = get_device()
    print(f"Device: {device}")

    SRC_VOCAB_SIZE = 10000
    TGT_VOCAB_SIZE = 10000
    EPOCHS = 20
    LR = 0.0001
    PAD_IDX = 0

    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    # 실제 데이터 로더 연결 필요
    # train_loader = DataLoader(...)
    # for epoch in range(1, EPOCHS + 1):
    #     loss = train_epoch(model, train_loader, criterion, optimizer, device, PAD_IDX)
    #     print(f"Epoch {epoch} | Loss: {loss:.4f}")

    print("Transformer 모델 정의 완료. 데이터셋 연결 필요.")


if __name__ == "__main__":
    main()
