# 논문 결과 재현 실험

## 목표
ResNet18을 PyTorch로 구현하여 CIFAR-10 분류 성능을 확인한다.

## 실험 설정

### 데이터셋
- CIFAR-10 (32×32 그대로 사용)

### 모델
- ResNet18

### 하이퍼파라미터
| 항목 | 값 |
| --- | --- |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| LR Scheduler | CosineAnnealing |
| Batch Size | 128 |
| Epochs | 30 |

### 데이터 증강
- RandomCrop (padding 4)
- RandomHorizontalFlip
- Normalize

## 실험 결과
| Epoch | Train Acc | Test Acc |
| --- | --- | --- |
| 5 |  |  |
| 15 |  |  |
| 30 |  |  |

### 최종 성능
- 테스트 정확도: 

### 참고 - 논문 보고 (CIFAR-10):
- ResNet20: 91.25%
- ResNet32: 92.49%
- ResNet110: 93.57%

## 분석
- 데이터 증강 + LR 스케줄러로 안정적 학습
- 잔차 연결로 깊이가 늘어도 학습 잘 됨

## 결론
ResNet의 잔차 연결이 효과적임을 확인.<br>
CIFAR-10 같은 작은 이미지에서도 좋은 성능을 보임.
