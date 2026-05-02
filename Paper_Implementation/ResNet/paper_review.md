# ResNet 논문 리뷰

## 기본 정보
- **제목**: Deep Residual Learning for Image Recognition
- **저자**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
- **발표**: CVPR 2016 (arxiv 2015), Best Paper
- **링크**: https://arxiv.org/abs/1512.03385

## 한 줄 요약
잔차 연결(Skip Connection)을 통해 매우 깊은 신경망의 학습을 가능하게 만든 혁신적인 논문.

## 배경
VGGNet 이후 "더 깊으면 더 좋다"는 트렌드.<br>
하지만 단순히 깊게 쌓으면 **degradation problem** 발생.

### Degradation Problem
- 56층이 20층보다 학습/테스트 오차 모두 더 높게 나옴
- 단순히 과적합 문제가 아님 (학습 오차도 높음)
- 깊어질수록 최적화가 어려워지는 현상

## 핵심 아이디어

### 잔차 학습 (Residual Learning)
신경망이 직접 $H(x)$를 학습하는 대신, **잔차** $F(x) = H(x) - x$를 학습한다.

$$H(x) = F(x) + x$$

```
일반:        H(x)        ← 학습
잔차:    F(x) + x        ← F(x)만 학습, x는 그대로 통과
         (학습)  (skip)
```

### 왜 잔차를 학습하는가?

**Identity mapping이 최적이라면:**
- 일반 방식: $H(x) = x$를 만들기 어려움 (가중치 정확히 맞춰야 함)
- 잔차 방식: $F(x) = 0$만 되면 됨 (모든 가중치 0이면 OK)

**기울기 소실 완화:**
- Skip connection으로 기울기가 직접 전달됨
- 매우 깊어도 기울기가 잘 흐름

## 모델 구조

### Basic Block (ResNet18, 34용)
```
입력 → Conv 3×3 → BN → ReLU → Conv 3×3 → BN → (+) → ReLU → 출력
                                              ↑
                                       Skip Connection
```

### Bottleneck Block (ResNet50+용)
```
입력 → Conv 1×1 → BN → ReLU → Conv 3×3 → BN → ReLU → Conv 1×1 → BN → (+) → ReLU
                                                                       ↑
                                                                Skip Connection
```
1×1로 채널 줄이고 → 3×3 연산 → 1×1로 채널 복원<br>
계산량 감소 + 깊이 증가

## 실험 결과 (ImageNet)
| 모델 | Top-5 에러 |
| --- | --- |
| VGG-19 | 7.3% |
| GoogLeNet | 6.7% |
| **ResNet-152** | **3.57%** |

→ 사람의 오차율 (5%)보다 낮은 최초의 모델

### 깊이별 성능
| 모델 | Top-5 에러 |
| --- | --- |
| ResNet-18 | 10.76% |
| ResNet-34 | 8.58% |
| ResNet-50 | 7.13% |
| ResNet-101 | 6.44% |
| ResNet-152 | 5.71% |

→ 깊어질수록 성능 향상 (VGG와 달리 한계 없음)

## 한계점

### 1. 여전히 큰 모델
ResNet152도 6천만 파라미터. 모바일/임베디드 환경에서는 가볍게 못 씀.<br>
→ MobileNet, EfficientNet 등으로 발전.

### 2. 추론 속도
깊이가 깊어 추론이 느림. Real-time 환경에선 부담.

## 느낀 점

### 1. 천재적인 아이디어
"Skip connection"이라는 단순한 변경 하나로 딥러닝 깊이의 한계를 깨뜨림.<br>
**가장 영향력 있는 딥러닝 논문 중 하나.**

### 2. 현대 모델의 표준
- Transformer에도 잔차 연결이 들어감
- DenseNet, EfficientNet 등 모든 현대 모델에 적용
- ResNet 이후 "잔차 연결 없는 깊은 모델"은 거의 없음

### 3. 깊이 vs 너비의 시작
ResNet 이후 "얼마나 깊게 vs 얼마나 넓게" 라는 트레이드오프 연구가 활발해짐.<br>
이는 EfficientNet의 compound scaling으로 이어짐.

## 면접 질문

**Q1: ResNet의 핵심 contribution은?**<br>
A: Skip connection을 통해 깊은 네트워크의 degradation problem을 해결하고,<br>
152층까지 학습 가능하게 만들어 ImageNet에서 사람보다 낮은 오차율을 달성했다.

**Q2: 왜 잔차를 학습하는 것이 효과적인가?**<br>
A: Identity mapping을 직접 학습하기는 어렵지만, 잔차를 0으로 만드는 건 쉽다.<br>
또한 skip connection으로 기울기가 직접 전달되어 기울기 소실 문제가 완화된다.

**Q3: BasicBlock과 Bottleneck의 차이는?**<br>
A: BasicBlock은 3×3 두 개로 단순. Bottleneck은 1×1 → 3×3 → 1×1로 채널을 줄였다 늘려 계산 효율↑.<br>
ResNet50 이상 깊은 모델에서 사용.

**Q4: 잔차 연결이 왜 작동한다고 생각하는가?**<br>
A: 1) Identity mapping 학습이 쉬움. 2) 기울기 흐름이 좋음. 3) 앙상블처럼 동작 (각 path가 다양한 깊이의 네트워크).
