# VGGNet PyTorch 구현

> **Very Deep Convolutional Networks for Large-Scale Image Recognition** (Simonyan & Zisserman, 2014)

## 개요
3×3 작은 필터를 깊게 쌓는 단순한 구조로 ImageNet에서 최고 성능을 보인 모델.<br>
VGG16, VGG19 등의 변종이 있다.

## 폴더 구조
```
VGGNet/
├── README.md
├── paper_review.md
├── model.py        # VGG11, 13, 16, 19 다 구현
├── train.py
├── experiments/
└── results/
```

## 모델 구조 (VGG16 기준)
```
Conv 64 - Conv 64 - Pool
Conv 128 - Conv 128 - Pool
Conv 256 - Conv 256 - Conv 256 - Pool
Conv 512 - Conv 512 - Conv 512 - Pool
Conv 512 - Conv 512 - Conv 512 - Pool
FC 4096 - FC 4096 - FC 1000
```
모든 Conv는 3×3, padding=1, stride=1<br>
모든 Pool은 2×2, stride=2

## 실행 방법
```bash
cd Paper_Implementation/VGGNet
python train.py
```

CIFAR-10으로 학습 (224×224 리사이즈).

## 핵심 contribution
1. **3×3 필터 깊게 쌓기** - 5×5, 7×7 큰 필터를 대체
2. **단순한 구조** - 같은 패턴(Conv-Conv-Pool)의 반복
3. **깊이의 중요성** 입증 - 16~19층까지 깊어져도 학습 가능

## 핵심 통찰
3×3 필터 두 개를 쌓으면 5×5 필터 하나와 같은 receptive field를 가지면서:
- 비선형성이 더 많이 들어감 (ReLU 두 번)
- 파라미터 수 더 적음 (2 × 3² = 18 < 5² = 25)

## 참고
- [원본 논문 (arxiv)](https://arxiv.org/abs/1409.1556)
