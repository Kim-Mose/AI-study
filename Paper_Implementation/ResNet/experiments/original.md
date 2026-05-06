# 논문 결과 재현 실험

## 목표
ResNet18을 PyTorch로 구현하여 CIFAR-10 분류 성능을 확인한다.

## 실험 설정

### 데이터셋
- CIFAR-10 (32×32 그대로 사용)
- 학습: 50,000장 (전체)
- 테스트: 10,000장

### 모델
- ResNet18 (BasicBlock × [2,2,2,2])

### 데이터 증강
- RandomCrop (padding 4)
- RandomHorizontalFlip
- Normalize

### 하이퍼파라미터
| 항목 | 값 |
| --- | --- |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| LR Scheduler | CosineAnnealingLR |
| Batch Size | 128 |
| Epochs | 5 (데모, 원본은 30+) |

### 환경
- Mac mini (MPS)
- 학습 시간: 약 2분 43초

## 실험 결과

### 학습 결과
| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
| --- | --- | --- | --- | --- |
| 1 | 1.5383 | 0.4386 | 1.3112 | 0.5327 |
| 2 | 1.1559 | 0.5889 | 0.9782 | 0.6578 |
| 3 | 0.9565 | 0.6615 | 0.9105 | 0.6889 |
| 4 | 0.8173 | 0.7109 | 0.7228 | 0.7470 |
| 5 | 0.7031 | 0.7514 | 0.6788 | 0.7619 |

### 최종 성능
- **테스트 정확도: 76.19% (5 epoch)**
- 참고 - 논문 보고 (CIFAR-10 ResNet20): 91.25%
- 30 epoch 학습 시 90% 이상 예상

## 분석

### 학습 안정성
- 잔차 연결(skip connection)과 BatchNorm 덕분에 매우 안정적인 학습
- 같은 깊이의 VGG가 학습 실패한 것과 대조적
- 첫 epoch부터 53% 달성 (랜덤 10% 대비 우수한 시작)

### 깊이의 효과 입증
- ResNet18은 18층의 깊은 네트워크임에도 안정적 학습
- 잔차 연결이 없었다면 학습이 어려웠을 것 (degradation problem)
- VGGNet 실험 실패와 비교하면 ResNet의 핵심 가치가 분명히 드러남

### 데이터 증강의 효과
- RandomCrop + RandomHorizontalFlip으로 일반화 성능 ↑
- Train과 Test 격차가 작음 → 과적합 적음

## 결론
ResNet18로 5 epoch만에 76% 달성. 충분한 epoch(30+)이면 90%+ 가능.<br>
**잔차 연결의 안정성**을 직접 확인. 깊은 네트워크 학습이 가능한 핵심 기술임을 검증.

## 다음 단계
- 30 epoch 전체 학습
- ResNet34, ResNet50과 깊이별 성능 비교
- Plain network와 비교 ablation
