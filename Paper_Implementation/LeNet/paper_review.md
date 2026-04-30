# LeNet-5 논문 리뷰

## 기본 정보
- **제목**: Gradient-Based Learning Applied to Document Recognition
- **저자**: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
- **발표**: Proceedings of the IEEE, 1998
- **링크**: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

## 한 줄 요약
손글씨 숫자 인식을 위해 합성곱 신경망(CNN)의 핵심 개념을 정립한 논문으로, 현대 CNN의 시초가 된 모델.

## 배경

### 기존 방식의 한계
1990년대까지의 패턴 인식은 **수작업 특성 추출 + 단순 분류기** 구조였다.
- 도메인 전문가가 직접 특성(feature)을 설계
- 이미지의 위치 변화, 크기 변화, 노이즈에 약함
- 새 문제마다 처음부터 다시 설계해야 함

### 왜 이 논문이 필요했나?
- 데이터가 많아지면서 자동으로 특성을 학습하는 방법이 필요
- 일반 MLP는 이미지를 펼쳐 1차원으로 처리하면 공간 정보 손실
- 파라미터 수가 너무 많아 학습이 어려움

## 핵심 아이디어

### 1. Local Receptive Field (지역 수용장)
이미지의 **인접한 픽셀끼리만** 연결하여 지역 패턴을 학습한다.
- 한 뉴런이 이미지 전체가 아닌 일부분만 봄
- 모서리, 끝점 등 지역적 특성을 잘 잡아냄

### 2. Weight Sharing (가중치 공유)
하나의 필터를 이미지 전체에 슬라이딩하며 적용한다.
- 파라미터 수 대폭 감소
- 위치에 상관없이 같은 패턴을 인식 (translation equivariance)

### 3. Subsampling (서브샘플링 / 풀링)
출력의 해상도를 줄여서 위치 변화에 강건하게 만든다.
- AvgPool 사용 (최근에는 MaxPool이 주류)
- 계산량 감소

## 모델 구조 (LeNet-5)

```
입력 32×32
   ↓ Conv 5×5 (6 filters)
C1: 28×28×6
   ↓ AvgPool 2×2
S2: 14×14×6
   ↓ Conv 5×5 (16 filters)
C3: 10×10×16
   ↓ AvgPool 2×2
S4: 5×5×16
   ↓ Conv 5×5 (120 filters)
C5: 1×1×120
   ↓ FC
F6: 84
   ↓ FC
출력: 10
```

활성화 함수: Sigmoid 또는 Tanh<br>
출력층: RBF (Radial Basis Function) - 현재는 Softmax + CE로 대체

## 학습 방법
- **데이터**: MNIST (60,000장 학습 + 10,000장 테스트)
- **손실 함수**: 최대우도 기반 (현재의 Cross-Entropy와 유사)
- **최적화**: Stochastic Gradient Descent
- **역전파**: 표준 backpropagation

## 실험 결과

| 모델 | 테스트 에러 |
| --- | --- |
| Linear Classifier | 12.0% |
| MLP (300-100) | 4.7% |
| **LeNet-5** | **0.95%** |
| LeNet-5 + 데이터 증강 | 0.8% |

→ MLP 대비 **5배** 이상 좋은 성능을 보임

## 한계점 및 개선 가능성

### 한계
1. **Sigmoid 활성화 함수** - 깊은 네트워크에서 기울기 소실 문제 발생
2. **AvgPool** - 노이즈에 약함 (MaxPool이 더 효과적인 경우 많음)
3. **작은 데이터셋** - MNIST는 비교적 단순한 데이터
4. **GPU 없는 시대** - 큰 모델 학습이 현실적으로 불가능

### 이후 발전
- **AlexNet (2012)** - ReLU, GPU 활용, Dropout 도입
- **VGGNet (2014)** - 작은 필터(3×3) 깊게 쌓기
- **ResNet (2015)** - 잔차 연결로 매우 깊은 신경망 학습 가능

## 느낀 점 / 인사이트

### 1. 본질적인 아이디어는 변하지 않았다
LeNet의 세 가지 핵심 원리(local receptive field, weight sharing, subsampling)는<br>
현대의 ResNet, EfficientNet 등에도 그대로 적용되어 있다. CNN의 본질을 이해하기에 가장 좋은 모델.

### 2. 시대적 한계
1998년 당시에는 CPU만으로 학습했고, 데이터도 부족했다.<br>
**좋은 아이디어가 있어도 하드웨어/데이터가 따라줘야 빛을 발한다**는 교훈.

### 3. 백본의 중요성
LeNet 이후 14년이 지나서야 AlexNet이 ImageNet에서 압도적 성능을 보여 딥러닝 부흥기가 시작됐다.<br>
연구의 가치는 **당장의 성능**보다 **이후 세대에 미치는 영향**으로 평가될 수 있다는 점이 흥미롭다.

## 면접 대비 질문

**Q1: LeNet의 핵심 contribution은?**<br>
A: CNN의 핵심 3원리(local receptive field, weight sharing, subsampling)를 정립하고<br>
역전파로 end-to-end 학습이 가능함을 실증했다.

**Q2: 왜 MLP 대신 CNN을 썼나?**<br>
A: MLP는 이미지를 1차원으로 펼쳐 공간 정보를 잃지만,<br>
CNN은 2차원 구조를 유지하면서 지역 패턴을 학습할 수 있다.<br>
또한 가중치 공유로 파라미터 수가 훨씬 적다.

**Q3: 현대 CNN과 LeNet의 차이는?**<br>
A: 활성화 함수(Sigmoid → ReLU), 풀링(AvgPool → MaxPool),<br>
정규화(없음 → BatchNorm), 깊이(7층 → 수십~수백 층),<br>
잔차 연결의 도입 등이 주요 차이점이다.

**Q4: 만약 본인이라면 LeNet을 어떻게 개선하겠는가?**<br>
A: ReLU 활성화로 기울기 소실 문제 해결, BatchNorm 추가,<br>
MaxPool 사용, Dropout으로 과적합 방지, 데이터 증강 등을 적용.<br>
이는 ablation 실험으로 검증할 수 있다.
