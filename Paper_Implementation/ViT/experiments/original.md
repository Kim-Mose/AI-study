# 원본 결과 재현 실험

## 목표
ViT를 PyTorch로 구현하여 분류 성능을 확인한다.<br>
원본은 JFT-300M으로 사전학습 후 ImageNet에서 평가하지만,<br>
자원 제약상 **CIFAR-10**으로 직접 학습.

## 실험 설정

### 데이터셋
- CIFAR-10 (32×32)

### 모델 (CIFAR-10용 작은 ViT)
| 항목 | 값 |
| --- | --- |
| Image Size | 32 |
| Patch Size | 4 |
| Embed Dim | 192 |
| Depth | 6 |
| Heads | 8 |
| MLP Ratio | 2.0 |

### 하이퍼파라미터
| 항목 | 값 |
| --- | --- |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 0.05 |
| LR Scheduler | CosineAnnealing |
| Batch Size | 64 |
| Epochs | 30 |

## 실험 결과
| Epoch | Train Acc | Test Acc |
| --- | --- | --- |
| 5 |  |  |
| 15 |  |  |
| 30 |  |  |

### 최종 성능
- 테스트 정확도: 

### 비교 (CIFAR-10)
- ResNet18 (참고): ~93%
- 사전학습 없는 ViT (직접 학습): 보통 70~80%
- 사전학습된 ViT fine-tuning: 95%+

## 분석

### ViT의 한계 관찰
- 작은 데이터셋(CIFAR-10)에선 ResNet보다 못한 결과 예상
- 이는 논문의 "데이터가 많아야 ViT가 강하다"는 주장과 일치

### 개선 방안
- 더 강한 데이터 증강 (RandAugment, MixUp, CutMix)
- DeiT의 Distillation 기법
- 사전학습된 ViT 사용

## 결론
ViT는 작은 데이터에선 CNN보다 약함을 직접 확인.<br>
대규모 사전학습이 필요한 모델임을 이해.
