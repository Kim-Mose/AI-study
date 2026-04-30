"""
BERT 학습 코드 템플릿
BERT는 두 가지 task로 사전학습된다.

1. MLM (Masked Language Model): 일부 토큰을 [MASK]로 가리고 예측
2. NSP (Next Sentence Prediction): 두 문장이 연속된 문장인지 분류

실제 사전학습은 데이터/시간이 매우 많이 필요해서
보통 사전학습된 모델(huggingface transformers)을 가져와서 fine-tuning한다.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from model import BERT, BERTForMLM, BERTForNSP


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def main():
    device = get_device()
    print(f"Device: {device}")

    VOCAB_SIZE = 30000
    EPOCHS = 10
    LR = 1e-4

    # 작은 BERT (실험용)
    bert = BERT(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
    ).to(device)

    mlm_model = BERTForMLM(bert, VOCAB_SIZE, d_model=128).to(device)
    nsp_model = BERTForNSP(bert, d_model=128).to(device)

    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(bert.parameters(), lr=LR)

    # 실제 데이터로 사전학습 또는 fine-tuning
    # for epoch in range(EPOCHS):
    #     for batch in train_loader:
    #         ...

    print("BERT 모델 정의 완료. 사전학습된 모델을 사용하거나 데이터셋 연결이 필요하다.")
    print("실무에서는 huggingface transformers 라이브러리 사용 권장")


if __name__ == "__main__":
    main()
