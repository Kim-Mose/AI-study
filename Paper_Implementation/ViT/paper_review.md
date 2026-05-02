# Vision Transformer (ViT) 논문 리뷰

## 기본 정보
- **제목**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **저자**: Alexey Dosovitskiy et al. (Google Research)
- **발표**: ICLR 2021 (arxiv 2020)
- **링크**: https://arxiv.org/abs/2010.11929

## 한 줄 요약
이미지를 패치로 나눠 Transformer로 처리하면 CNN 없이도 SOTA를 달성할 수 있음을 보인 논문.

## 배경

### 비전 분야의 통념
- "이미지에는 CNN이 최적"
- CNN의 **inductive bias** (지역성, 변환 불변성)이 이미지 처리에 유리

### Transformer의 부상
- NLP에서 Transformer가 모든 task를 평정
- "이미지에도 Transformer만으로 충분할까?"

## 핵심 아이디어

### 1. 이미지를 패치로 나누기
이미지를 16×16 패치로 분할하여 단어처럼 처리.

```
224×224 이미지 → 16×16 패치 × 196개
각 패치 (16×16×3=768) → 임베딩 (D 차원)
```

### 2. 표준 Transformer 인코더 사용
- BERT와 거의 같은 구조
- Self-Attention + Feed Forward

### 3. [CLS] 토큰
BERT처럼 분류용 [CLS] 토큰 추가 → 마지막 [CLS] 출력으로 분류

### 4. Positional Embedding
학습 가능한 1D positional embedding 사용 (2D나 sinusoidal도 시도, 차이 미미)

## 모델 구조

```
이미지 (H×W×C)
   ↓ Patch & Flatten
패치 (N × P²·C)  N = HW/P²
   ↓ Linear Projection
패치 임베딩 (N × D)
   ↓ + [CLS] 토큰 → (N+1 × D)
   ↓ + Positional Embedding
   ↓
Transformer Encoder Block × L
├── LayerNorm
├── Multi-Head Self-Attention
├── + Residual
├── LayerNorm
├── MLP
└── + Residual
   ↓
LayerNorm
   ↓
[CLS] 출력 → MLP Head → 클래스
```

## 실험 결과

### ImageNet 성능
| 모델 | Top-1 |
| --- | --- |
| ResNet-152x4 | 87.54 |
| EfficientNet-B7 | 88.4 |
| **ViT-H/14 (JFT 사전학습)** | **88.55** |

### 핵심 발견: 데이터 양의 중요성
| 사전학습 데이터 | 작은 모델 | 큰 모델 |
| --- | --- | --- |
| ImageNet (1.3M) | CNN > ViT | CNN ≥ ViT |
| ImageNet-21k (14M) | CNN ≈ ViT | ViT > CNN |
| JFT-300M (300M) | ViT > CNN | ViT >> CNN |

→ **큰 데이터셋이 있어야 ViT가 빛을 발함**

## 핵심 통찰

### Inductive Bias의 trade-off
- CNN: 강한 inductive bias → 적은 데이터로도 잘 됨
- Transformer: 약한 inductive bias → 데이터 많으면 더 강해짐

### 즉, 데이터가 많으면 학습으로 inductive bias를 만들 수 있다.

## 한계점

### 1. 데이터 의존성
작은 데이터셋(CIFAR-10 등)에서는 CNN보다 못함.<br>
대규모 사전학습 필요.

### 2. 계산량
글로벌 어텐션은 O(N²). 큰 이미지에서 부담.<br>
→ Swin Transformer 등이 윈도우 어텐션으로 해결.

### 3. 위치 정보의 한계
1D positional embedding은 2D 이미지의 공간 정보를 완벽히 표현 못함.

## 후속 연구
- **DeiT (2021)**: ViT를 ImageNet만으로 학습 가능하게 (Distillation)
- **Swin Transformer**: 계층적 구조 + Shifted Window Attention
- **MAE**: Masked Autoencoder로 사전학습
- **CLIP**: 텍스트-이미지 멀티모달

## 느낀 점

### 1. CV와 NLP의 통합
같은 Transformer 구조로 양쪽 분야 모두 처리 가능.<br>
멀티모달 모델(CLIP, DALL-E 등)의 토대가 됨.

### 2. 데이터 스케일의 중요성
Inductive bias가 약해도 데이터가 많으면 괜찮음.<br>
이는 LLM의 scaling law와도 통하는 관찰.

### 3. 단순함의 미학
복잡한 CNN 구조 없이 단순한 패치 분할 + Transformer로 SOTA.

## 면접 질문

**Q1: ViT의 핵심 아이디어는?**<br>
A: 이미지를 패치로 나눠 단어처럼 취급하고, 표준 Transformer 인코더로 처리.<br>
CNN 없이도 SOTA가 가능함을 입증.

**Q2: 왜 큰 데이터셋이 필요한가?**<br>
A: ViT는 CNN보다 inductive bias(지역성, 변환 불변성)가 약하다.<br>
큰 데이터로 학습해야 이런 패턴을 데이터에서 학습할 수 있다.

**Q3: ViT와 CNN의 trade-off는?**<br>
A: CNN은 적은 데이터에 강하지만 long-range dependency가 약함.<br>
ViT는 큰 데이터가 있으면 더 강하지만 작은 데이터에선 약함.

**Q4: ViT의 [CLS] 토큰 역할은?**<br>
A: BERT처럼 분류용 토큰. 모든 패치의 정보를 종합하여 마지막에 분류 헤드로 전달.

**Q5: Patch Size가 작을수록 좋은가?**<br>
A: 작을수록 더 많은 패치 → 표현력 ↑, 하지만 계산량 ↑ (O(N²)).<br>
보통 16×16이 균형점.
