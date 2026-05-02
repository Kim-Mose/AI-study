# Vision Transformer (ViT) PyTorch 구현

> **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (Dosovitskiy et al., 2020)

## 개요
이미지를 패치로 나눠 Transformer로 처리한 모델.<br>
"이미지에도 Transformer만으로 충분하다"는 것을 입증한 논문.

## 폴더 구조
```
ViT/
├── README.md
├── paper_review.md
├── model.py
├── train.py
├── experiments/
└── results/
```

## 핵심 아이디어
이미지를 16×16 패치로 분할 → 각 패치를 단어처럼 처리 → Transformer 인코더 통과

```
이미지 (224×224×3) → 패치 (16×16×3) × 196개
   ↓ Linear Projection
패치 임베딩 (196 × D)
   ↓ + [CLS] 토큰 + Positional Encoding
   ↓
Transformer Encoder × N
   ↓
[CLS] 출력 → MLP → 클래스
```

## 모델 변종
| 모델 | Layers | Hidden | Heads | Params |
| --- | --- | --- | --- | --- |
| ViT-Base | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1024 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 16 | 632M |

## 실행 방법
```bash
cd Paper_Implementation/ViT
python train.py
```

CIFAR-10용 작은 ViT로 학습.

## 핵심 contribution
1. **CNN 없이 이미지 처리** - 어텐션만으로 SOTA
2. **확장성** - 데이터/모델 크기 늘릴수록 성능 ↑
3. **NLP와 CV의 통합** - 같은 Transformer 구조 사용

## 한계 및 고려사항
- **데이터 의존성** - 작은 데이터셋에서는 CNN보다 약함
- **사전학습 필요** - JFT-300M 같은 큰 데이터셋으로 사전학습해야 진가 발휘

## 참고
- [원본 논문 (arxiv)](https://arxiv.org/abs/2010.11929)
