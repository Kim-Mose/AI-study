# 원본 결과 재현 실험

## 목표
ViT를 PyTorch로 구현하여 분류 성능을 확인한다.<br>
원본은 JFT-300M으로 사전학습 후 ImageNet에서 평가하지만,<br>
자원 제약상 **CIFAR-10**으로 직접 학습.

## 실험 설정

### 데이터셋
- CIFAR-10 (32×32)
- 학습: 50,000장 (전체)
- 테스트: 10,000장

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
| Learning Rate | 0.0003 |
| Weight Decay | 0.05 |
| LR Scheduler | CosineAnnealingLR |
| Batch Size | 64 |
| Epochs | 5 (데모, 원본은 30+) |

### 환경
- Mac mini (MPS)

## 실험 결과

### 학습 결과
| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
| --- | --- | --- | --- | --- |
| 1 | 1.6408 | 0.3932 | 1.3448 | 0.5091 |
| 2 | 1.2551 | 0.5454 | 1.1935 | 0.5652 |
| 3 | 1.0904 | 0.6056 | 1.1121 | 0.5891 |
| 4 | 0.9544 | 0.6550 | 1.0519 | 0.6242 |
| 5 | 0.8314 | 0.7043 | 1.0001 | 0.6449 |

### 최종 성능
- **테스트 정확도: 64.49% (5 epoch)**

### 비교 (CIFAR-10)
- ResNet18 (5 epoch): **76.19%**
- ViT (5 epoch): **64.49%** ← 본 결과
- 사전학습된 ViT fine-tuning: 95%+

## 분석

### ViT < ResNet (작은 데이터셋에서)
- 같은 epoch에서 ResNet18(76%)이 ViT(64.5%)보다 12%p 높음
- 이는 논문의 주장 "**ViT는 큰 데이터에서 빛난다**"와 정확히 일치
- CNN의 inductive bias (지역성, 평행이동 불변성)가 작은 데이터에 유리

### 학습 진행 상황
- 첫 epoch부터 39% (랜덤 10% 대비 높음)
- 학습은 잘 진행되나 ResNet보다 느림
- 더 많은 epoch과 강한 데이터 증강이 필요

### 이론과 실험의 일치
- 논문 결과: JFT-300M 사전학습 시 ViT > CNN
- 본 실험: 작은 데이터셋에서 ViT < CNN
- ViT의 한계와 강점을 직접 체험

## 개선 방안
1. **강한 데이터 증강** (RandAugment, MixUp, CutMix)
2. **DeiT의 Distillation Token**
3. **사전학습 모델 사용** (timm 라이브러리)
4. **더 많은 epoch** (50+)

## 결론
ViT가 CIFAR-10 같은 작은 데이터셋에서 CNN보다 약하다는 점을 직접 검증.<br>
이는 inductive bias의 가치를 보여주는 좋은 예시이며,<br>
**ViT의 진가는 큰 사전학습 데이터에서 발휘됨**을 확인.
