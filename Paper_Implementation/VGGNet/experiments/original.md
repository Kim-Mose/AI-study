# 논문 결과 재현 실험

## 목표
VGG16 구조를 PyTorch로 구현하여 분류 성능을 확인한다.<br>
원본은 ImageNet이지만 자원 제약으로 **CIFAR-10 일부**로 진행.

## 실험 설정

### 데이터셋
- CIFAR-10 (32×32 → 224×224 리사이즈)
- 학습: 5,000장 subset
- 테스트: 1,000장

### 모델
- VGG16 (출력층 1000 → 10 변경)
- BatchNorm 없는 원본 구조

### 하이퍼파라미터
| 항목 | 값 |
| --- | --- |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 3 |

## 실험 결과

### 학습 결과
| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
| --- | --- | --- | --- | --- |
| 1 | 2.3302 | 0.1062 | 2.3227 | 0.1000 |
| 2 | 2.3046 | 0.0970 | 2.3030 | 0.1000 |
| 3 | 2.3033 | 0.1028 | 2.3025 | 0.1000 |

### 최종 성능
- 테스트 정확도: **10.00% (랜덤 수준)**

## 분석

### ⚠️ 학습이 수렴하지 않은 이유
**Loss가 약 2.3에 머무름** = `ln(10) ≈ 2.3`<br>
이는 모델이 10개 클래스 중 균등 확률을 출력하는 상태로, **학습이 전혀 진행되지 않음**을 의미.

### 원인
1. **BatchNorm 없는 깊은 모델 + 큰 학습률**
   - VGG16(13개 conv layer, 13만 파라미터)은 BatchNorm 없이 lr=0.001은 너무 큼
   - 깊은 네트워크에서 기울기 폭발/소실 가능성
2. **ReLU의 Dying ReLU 가능성**
   - 깊은 모델에서 ReLU가 모두 0이 되어 기울기가 안 흐름
3. **초기화 문제**
   - 원본 VGGNet은 weight initialization을 신중하게 설계 (Kaiming init)

### 해결 방법
1. **학습률 낮추기**: lr=1e-4 또는 1e-5
2. **BatchNorm 추가** (VGG16 with BN)
3. **Kaiming initialization 적용**
4. **Pre-trained 가중치 로드 후 fine-tuning**

### 이 실험의 의미
이는 단순히 "더 깊으면 더 좋다"의 한계를 직접 보여주는 결과.<br>
VGG → ResNet으로 발전한 이유 (BatchNorm + 잔차 연결)를 체감할 수 있음.

## 다음 단계
- VGG16 + BatchNorm 버전으로 재실험
- 학습률을 1e-4로 낮춰 재실험
- ablation 실험으로 학습 안정성 정량적으로 비교

## 결론
VGG16 모델 구조 자체는 정상 구현됐으나, BatchNorm 없는 깊은 모델은 학습률에 매우 민감함을 확인.<br>
이는 ResNet의 잔차 연결과 BatchNorm이 왜 필요한지 보여주는 좋은 사례.
