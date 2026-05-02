# VGGNet 논문 리뷰

## 기본 정보
- **제목**: Very Deep Convolutional Networks for Large-Scale Image Recognition
- **저자**: Karen Simonyan, Andrew Zisserman (Oxford VGG)
- **발표**: ICLR 2015 (arxiv 2014)
- **링크**: https://arxiv.org/abs/1409.1556

## 한 줄 요약
3×3 작은 필터를 단순하게 깊이 쌓는 것만으로 ImageNet에서 최고 수준의 성능을 달성한 논문.

## 배경
AlexNet 이후 더 깊고 좋은 모델이 필요했다.
- AlexNet: 11×11, 5×5 등 다양한 필터 크기 사용
- 어떻게 하면 효과적으로 더 깊은 모델을 만들 수 있을까?

## 핵심 아이디어

### 1. 3×3 필터의 효율성
3×3 필터 N개를 쌓으면 receptive field가 (2N+1)×(2N+1)이 된다.
- 3×3 두 개 = 5×5 한 개와 같은 영역 커버
- 3×3 세 개 = 7×7 한 개와 같은 영역 커버

**3×3을 여러 개 쌓는 게 더 좋은 이유:**
1. **비선형성 증가** - 사이마다 ReLU 들어가서 표현력 ↑
2. **파라미터 감소** - 3×(3²) = 27 < 7² = 49

### 2. 깊이의 중요성
같은 컴퓨팅 비용으로 깊게 쌓으면 성능이 향상됨을 실증.<br>
VGG11, VGG13, VGG16, VGG19로 갈수록 성능 향상.

### 3. 단순하고 균일한 구조
모든 Conv = 3×3, stride 1, padding 1<br>
모든 Pool = 2×2, stride 2

→ 단순해서 분석/구현/응용이 쉽다.

## 모델 구조

| 모델 | Conv 층 수 | 총 층 수 |
| --- | --- | --- |
| VGG11 | 8 | 11 |
| VGG13 | 10 | 13 |
| **VGG16** | 13 | 16 |
| VGG19 | 16 | 19 |

VGG16 구조:
```
[Conv64 - Conv64] - Pool
[Conv128 - Conv128] - Pool
[Conv256 - Conv256 - Conv256] - Pool
[Conv512 - Conv512 - Conv512] - Pool
[Conv512 - Conv512 - Conv512] - Pool
FC4096 - FC4096 - FC1000
```

## 실험 결과 (ImageNet)
| 모델 | Top-5 에러 |
| --- | --- |
| AlexNet | 15.3% |
| VGG16 | 7.3% |
| VGG19 | 7.0% |

→ AlexNet의 절반 수준 에러

## 한계점

### 1. 파라미터 수가 많음
VGG16 = 약 1억 3,800만 파라미터. 대부분 FC 층에 있음.

### 2. 학습 시간
4개의 NVIDIA Titan GPU로 2~3주 학습.

### 3. 이후 등장한 ResNet
VGG도 깊어질수록 성능 한계 보임 → ResNet의 잔차 연결로 극복.

## 느낀 점

### 1. 단순함의 미학
복잡한 트릭 없이 **3×3 필터 + 깊이**라는 단순한 원칙으로 SOTA를 달성.<br>
"좋은 모델은 단순해야 한다"는 좋은 예시.

### 2. 깊이의 한계 시사
VGG19 이상으로 깊어지면 오히려 성능 하락.<br>
이 문제가 ResNet의 잔차 연결로 해결됨.

### 3. Pre-trained Backbone의 시작
VGG는 ImageNet으로 학습한 후 다른 task의 backbone으로 널리 쓰였다.<br>
**Transfer Learning** 시대를 연 모델.

## 면접 질문

**Q1: 왜 3×3 필터를 사용하나?**<br>
A: 같은 receptive field를 더 적은 파라미터와 더 많은 비선형성으로 만들 수 있기 때문.<br>
3×3 두 개 = 5×5 하나, 파라미터는 18 < 25.

**Q2: VGG와 AlexNet의 차이는?**<br>
A: AlexNet은 다양한 크기의 필터 + LRN 사용. VGG는 3×3 필터만 사용 + 더 깊음.<br>
VGG가 더 단순하고 성능도 좋음.

**Q3: VGG의 한계는?**<br>
A: 파라미터가 많아 메모리/계산 비용이 큼. 더 깊어지면 학습이 어려움 (ResNet으로 해결).
