# Transformer PyTorch 구현

> **Attention Is All You Need** (Vaswani et al., 2017)

## 개요
RNN을 사용하지 않고 오직 어텐션만으로 시퀀스를 처리하는 혁신적인 모델.<br>
GPT, BERT, T5 등 현대 모든 LLM의 기반이 되는 구조.

## 폴더 구조
```
Transformer/
├── README.md
├── paper_review.md
├── model.py
├── train.py        # 번역 학습 템플릿
├── experiments/
└── results/
```

## 모델 구조
```
인코더 (N=6)
├── Multi-Head Self-Attention
├── Add & LayerNorm
├── Feed Forward
└── Add & LayerNorm

디코더 (N=6)
├── Masked Multi-Head Self-Attention
├── Add & LayerNorm
├── Cross-Attention (Encoder-Decoder)
├── Add & LayerNorm
├── Feed Forward
└── Add & LayerNorm
```

## 핵심 컴포넌트
1. **Scaled Dot-Product Attention**
2. **Multi-Head Attention**
3. **Positional Encoding** (sin, cos)
4. **Layer Normalization**
5. **Residual Connection**

## 실행 방법
```bash
cd Paper_Implementation/Transformer
python model.py  # 모델 동작 확인
```

학습은 번역 데이터셋 필요 (WMT, IWSLT 등).

## 핵심 수식
$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

## 핵심 contribution
1. **RNN 없이 시퀀스 처리** - 병렬 학습 가능
2. **장기 의존성 해결** - 어텐션으로 모든 위치 직접 참조
3. **확장성** - 더 깊고 더 큰 모델로 쉽게 확장
4. **현대 LLM의 기반**

## 참고
- [원본 논문 (arxiv)](https://arxiv.org/abs/1706.03762)
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
