# ResNet PyTorch 구현

> **Deep Residual Learning for Image Recognition** (He et al., 2015)

## 개요
잔차 연결(Skip Connection)로 매우 깊은 신경망(50, 101, 152층)도 학습 가능하게 만든 모델.<br>
ImageNet 2015 우승작이며, 현재까지도 가장 영향력 있는 논문 중 하나.

## 폴더 구조
```
ResNet/
├── README.md
├── paper_review.md
├── model.py        # ResNet18, 34, 50, 101, 152
├── train.py
├── experiments/
└── results/
```

## 주요 변종
| 모델 | 블록 | 층 수 |
| --- | --- | --- |
| ResNet18 | BasicBlock × [2,2,2,2] | 18 |
| ResNet34 | BasicBlock × [3,4,6,3] | 34 |
| ResNet50 | Bottleneck × [3,4,6,3] | 50 |
| ResNet101 | Bottleneck × [3,4,23,3] | 101 |
| ResNet152 | Bottleneck × [3,8,36,3] | 152 |

## 잔차 블록 (Residual Block)
```
입력 x
  ├──→ Conv → BN → ReLU → Conv → BN → +
  └──────────────────────── Skip ────→
                                       ↓
                                     ReLU
```
$y = F(x) + x$

## 실행 방법
```bash
cd Paper_Implementation/ResNet
python train.py
```

CIFAR-10으로 학습. 기본은 ResNet18.

## 핵심 contribution
1. **잔차 연결** - 깊어질수록 성능이 떨어지는 문제(degradation problem) 해결
2. **152층까지 학습 가능** - 이전엔 불가능했던 깊이
3. **Bottleneck 구조** - 1×1 → 3×3 → 1×1 으로 효율적 계산

## 참고
- [원본 논문 (arxiv)](https://arxiv.org/abs/1512.03385)
