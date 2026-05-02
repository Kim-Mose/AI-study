# Transformer 논문 리뷰

## 기본 정보
- **제목**: Attention Is All You Need
- **저자**: Ashish Vaswani et al. (Google Brain)
- **발표**: NeurIPS 2017
- **링크**: https://arxiv.org/abs/1706.03762

## 한 줄 요약
RNN/CNN 없이 어텐션만으로 시퀀스를 처리하는 모델로, 현대 LLM의 기반이 된 가장 영향력 있는 논문 중 하나.

## 배경

### 기존 시퀀스 모델의 문제
- **RNN/LSTM**: 순차 처리 → 병렬화 어려움, 학습 느림
- **CNN**: 거리에 따라 받는 영향이 제한적
- 둘 다 장기 의존성 학습이 어려움

### 어텐션의 등장
- 이전에도 어텐션은 있었지만 RNN의 보조로만 사용
- "어텐션만으로도 충분하지 않을까?"

## 핵심 아이디어

### 1. Self-Attention
시퀀스 내 모든 위치를 직접 참조.

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| 용어 | 역할 |
| --- | --- |
| Query (Q) | 무엇을 찾는가 |
| Key (K) | 각 위치의 특징 |
| Value (V) | 실제 값 |

$\sqrt{d_k}$로 나누는 이유: 차원이 클 때 내적값이 너무 커져 softmax가 saturate되는 것 방지.

### 2. Multi-Head Attention
어텐션을 여러 번 병렬로 수행해 다양한 관점에서 관계 학습.

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$$

논문에서는 h=8.

### 3. Positional Encoding
RNN과 달리 순서 정보가 없으므로 직접 부여.

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

### 4. Encoder-Decoder 구조
- 인코더: 입력 시퀀스를 표현
- 디코더: Masked Self-Attention + Cross-Attention으로 출력 생성

## 모델 구조

```
[입력] → 임베딩 + Positional Encoding
   ↓
인코더 × 6
├── Multi-Head Self-Attention
├── Add & Norm
├── Feed Forward
└── Add & Norm
   ↓
디코더 × 6
├── Masked Multi-Head Self-Attention
├── Add & Norm
├── Multi-Head Cross-Attention
├── Add & Norm
├── Feed Forward
└── Add & Norm
   ↓
Linear + Softmax → [출력]
```

## 실험 결과

### 기계 번역 (WMT 2014 영어-독일어)
| 모델 | BLEU |
| --- | --- |
| ByteNet | 23.75 |
| GNMT | 24.6 |
| ConvS2S | 25.16 |
| **Transformer (big)** | **28.4** |

→ 기존 SOTA 대비 큰 향상, 학습 시간도 1/4

## 한계점

### 1. O(n²) 복잡도
시퀀스 길이의 제곱에 비례하는 메모리/계산.<br>
긴 시퀀스에서 부담 → Longformer, Linformer 등이 해결 시도.

### 2. 학습 데이터 의존성
큰 데이터가 있어야 좋은 성능. 작은 데이터셋에서는 RNN보다 못한 경우도.

### 3. Positional Encoding의 한계
학습 시 본 길이 이상의 시퀀스에 약함.<br>
→ Relative Positional Encoding, RoPE 등으로 발전.

## 느낀 점

### 1. 현대 AI의 기반
- BERT, GPT, T5, ViT, CLIP 모두 Transformer 기반
- "트랜스포머 = 현대 딥러닝"이라 해도 과언 아님

### 2. 단순함의 위력
복잡한 RNN 구조 제거하고 어텐션만으로 더 좋은 성능.<br>
"Less is more"의 좋은 예.

### 3. 확장성
파라미터/데이터를 키울수록 성능이 계속 향상 (LLM의 scaling law).<br>
이는 GPT-3, GPT-4로 이어짐.

## 면접 질문

**Q1: Transformer의 핵심 contribution은?**<br>
A: RNN/CNN 없이 어텐션만으로 시퀀스를 처리할 수 있음을 보였다.<br>
병렬 학습 가능, 장기 의존성 해결, 확장성이 좋아 현대 LLM의 기반이 됐다.

**Q2: Self-Attention과 Cross-Attention의 차이?**<br>
A: Self-Attention은 같은 시퀀스 내에서 Q, K, V가 모두 나옴.<br>
Cross-Attention은 디코더에서 Q는 디코더, K/V는 인코더에서 나옴.

**Q3: 왜 √d_k로 나누나?**<br>
A: 차원이 크면 내적값이 매우 커져 softmax가 saturate(한쪽으로 쏠림)되어<br>
기울기가 사라진다. √d_k로 나눠 적절한 분포 유지.

**Q4: Positional Encoding이 왜 필요한가?**<br>
A: 어텐션은 순서를 인식하지 못한다. 위치 정보를 명시적으로 더해줘야<br>
"나는 학교에 간다"와 "학교는 나를 간다"를 구분할 수 있다.

**Q5: Multi-Head를 왜 쓰나?**<br>
A: 여러 헤드가 서로 다른 표현 공간에서 정보를 처리.<br>
한 헤드는 문법, 다른 헤드는 의미 등 다양한 관점 학습 가능.
