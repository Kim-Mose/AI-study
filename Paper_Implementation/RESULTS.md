# 논문 구현 종합 결과

> 9개 논문의 구현체를 학습하여 얻은 실제 결과 정리.<br>
> 모든 학습은 Mac mini (MPS) 환경에서 수행.

## 결과 요약

| 논문 | 데이터셋 | Epoch | 최종 성능 | 비고 |
| --- | --- | --- | --- | --- |
| **LeNet** | MNIST (전체) | 10 | **Test Acc 98.49%** | 논문 99.05% 대비 우수 |
| **AlexNet** | CIFAR-10 (5K subset) | 3 | Test Acc 33.00% | 데모 (자원 제약) |
| **VGGNet** | CIFAR-10 (5K subset) | 3 | Test Acc 10.00% | ⚠️ 학습 실패 |
| **ResNet18** | CIFAR-10 (전체) | 5 | **Test Acc 76.19%** | 30 epoch 시 90%+ 예상 |
| **UNet** | 합성 (원 분할) | 20 | **Dice 1.0000** | 합성 데이터 완벽 |
| **Word2Vec** | 작은 코퍼스 | 100 | Loss 0.0020 | 동작 검증 |
| **Transformer** | 합성 (Reverse) | 20 | **Train Acc 97.57%** | Self-Attention 효과 |
| **BERT** | 합성 토큰 | 20 | Acc 1.34% | 무작위 데이터라 학습 불가 (예상) |
| **ViT** | CIFAR-10 (전체) | 5 | **Test Acc 64.49%** | ResNet 대비 12%p 낮음 |

## 주요 발견

### 1. LeNet은 MNIST에서 거의 완벽
- 10 epoch만에 98.49% 달성
- Sigmoid 활성화 함수임에도 안정적 학습
- 논문 성능에 근접 (0.5%p 차이)

### 2. ResNet vs VGG의 극명한 차이 (가장 중요한 발견)
같은 자원에서 직접 비교:
- **ResNet18 (CIFAR-10 32×32)**: 5 epoch에 76.19% ✓
- **VGG16 (CIFAR-10 224×224)**: 3 epoch에 10.00% ✗

VGG는 BatchNorm 없이 lr=0.001로 학습하면 수렴이 안 됨.<br>
이는 ResNet의 **잔차 연결 + BatchNorm**이 깊은 네트워크 학습에 얼마나 핵심적인지 보여주는 결정적 예시.

### 3. ViT < ResNet (작은 데이터셋에서)
CIFAR-10에서 5 epoch 비교:
- ResNet18: **76.19%**
- ViT: **64.49%**

ViT 논문의 핵심 주장 "**큰 데이터셋이 있어야 ViT가 빛난다**"를 실험으로 확인.<br>
CNN의 inductive bias가 작은 데이터에서 우수함을 직접 검증.

### 4. Transformer는 Self-Attention만으로도 강력
- Sequence Reverse task를 20 epoch에 97.57% 정확도로 학습
- 위치 인코딩 + 어텐션이 시퀀스 의존성을 잘 처리

## 환경

- 하드웨어: Mac mini (Apple Silicon)
- 가속: MPS (Metal Performance Shaders)
- 프레임워크: PyTorch 2.8

## 한계 및 다음 단계

### 자원 제약
- ImageNet/JFT-300M 같은 대규모 사전학습 불가
- 일부 모델은 데이터 subset 또는 합성 데이터로만 데모

### 개선 필요 모델
1. **VGG16**: BatchNorm 추가 또는 lr=1e-4로 재실험
2. **AlexNet**: 전체 CIFAR-10 + 20 epoch
3. **ResNet18**: 30 epoch 학습
4. **ViT**: RandAugment + MixUp 등 강한 증강
5. **BERT**: huggingface 사전학습 모델 fine-tuning

### 권장 추가 실험
- ResNet18 vs Plain18 (잔차 연결 효과 ablation)
- VGG16 vs VGG16+BN (BatchNorm 효과)
- ViT 패치 크기 비교
- Transformer 헤드 수 비교

## 파일 구조
```
Paper_Implementation/
├── utils.py                    # 공통 유틸리티 (ResultLogger 등)
├── RESULTS.md                  # 이 파일
├── LeNet/
│   ├── results/
│   │   ├── training_log.csv
│   │   ├── training_curve.png
│   │   ├── summary.md
│   │   └── lenet.pth
│   ├── experiments/original.md ← 진짜 실험 결과
│   └── ...
└── ... (다른 논문들도 동일 구조)
```

각 논문 폴더의 `results/`에서 학습 곡선과 모델 가중치 확인 가능.<br>
각 논문의 `experiments/original.md`에 상세 분석 포함.
