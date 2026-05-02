# AlexNet 논문 리뷰

## 기본 정보
- **제목**: ImageNet Classification with Deep Convolutional Neural Networks
- **저자**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **발표**: NeurIPS 2012
- **링크**: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

## 한 줄 요약
대규모 GPU 학습과 ReLU, Dropout 등을 결합하여 ImageNet에서 압도적 성능을 보이며 딥러닝 시대를 연 논문.

## 배경
2010~2011년 ImageNet 우승작은 SIFT 같은 수작업 특성 + SVM 조합이었다.<br>
2010년 우승: 28.2% 에러, 2011년 우승: 25.8% 에러.<br>
LeNet은 있었지만, 큰 데이터셋과 깊은 모델을 학습할 하드웨어가 부족했다.

## 핵심 아이디어

### 1. ReLU 활성화 함수
$$f(x) = \max(0, x)$$
- Sigmoid/Tanh보다 학습 속도 6배 빠름
- 기울기 소실 문제 완화
- 단순해서 계산량 적음

### 2. GPU 학습
- 2개의 NVIDIA GTX 580 GPU 사용 (각 3GB 메모리)
- 모델을 두 GPU에 분산 (Model Parallelism)
- 5~6일 학습

### 3. Local Response Normalization (LRN)
같은 위치의 인접 채널 간 정규화. 현재는 BatchNorm이 더 효과적이라 잘 안 씀.

### 4. Overlapping Pooling
풀링 스트라이드가 풀링 크기보다 작음 (2×2 윈도우, stride 2 → 3×3 윈도우, stride 2)

### 5. Dropout
FC 층에서 0.5 확률로 뉴런 제거 → 과적합 방지

### 6. 데이터 증강
- 랜덤 224×224 크롭
- 좌우 반전
- PCA 기반 컬러 변형

## 모델 구조
```
입력: 224×224×3
Conv1 (96, 11×11, stride 4) + ReLU + LRN + MaxPool 3×3 stride 2
Conv2 (256, 5×5) + ReLU + LRN + MaxPool
Conv3 (384, 3×3) + ReLU
Conv4 (384, 3×3) + ReLU
Conv5 (256, 3×3) + ReLU + MaxPool
FC (4096) + ReLU + Dropout
FC (4096) + ReLU + Dropout
FC (1000)
```
파라미터: 약 6,000만개

## 실험 결과 (ImageNet)
| 모델 | Top-5 에러 |
| --- | --- |
| 2010 우승 | 28.2% |
| 2011 우승 | 25.8% |
| **AlexNet** | **15.3%** |

→ 이전 우승작보다 **10%p 이상** 향상

## 한계점 및 개선 가능성
1. **LRN의 효과** - 이후 BatchNorm으로 대체됨
2. **큰 필터 (11×11)** - VGGNet에서 작은 필터(3×3) 깊게 쌓는 게 더 효과적임을 입증
3. **모델이 큼** - 6천만 파라미터. 임베디드 환경 부적합

## 느낀 점

### 1. ReLU의 위대함
간단한 변경(Sigmoid → ReLU)이 딥러닝 학습을 근본적으로 가능하게 만들었다.<br>
**때로는 단순한 아이디어가 가장 강력하다.**

### 2. 하드웨어와 알고리즘의 동행
GPU가 없었다면 AlexNet도 빛을 못 봤을 것.<br>
연구 아이디어와 컴퓨팅 자원의 결합이 만든 혁명.

### 3. End-to-end 학습의 시작
이전: 수작업 특성 추출 + 분류기<br>
AlexNet 이후: 입력에서 출력까지 신경망이 전부 처리

## 면접 질문

**Q1: AlexNet이 LeNet과 다른 점?**<br>
A: 더 깊고 (5 conv layers), ReLU 사용, GPU 학습, Dropout, 데이터 증강 등 현대 딥러닝의 토대를 만들었다.

**Q2: ReLU의 장단점은?**<br>
A: 장점 - 빠른 학습, 기울기 소실 완화, 계산 단순.<br>
단점 - "Dying ReLU" (음수에서 기울기 0). Leaky ReLU, GELU 등이 보완.

**Q3: 왜 모델을 두 GPU에 나눴나?**<br>
A: 당시 GPU 메모리(3GB)가 부족해서 단일 GPU에 모델이 안 들어갔기 때문.<br>
지금은 메모리가 충분해 잘 안 쓰는 기법.
