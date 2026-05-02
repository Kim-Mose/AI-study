# Word2Vec PyTorch 구현

> **Efficient Estimation of Word Representations in Vector Space** (Mikolov et al., 2013)

## 개요
단어를 고정된 차원의 벡터로 표현하는 단어 임베딩 모델.<br>
NLP의 입력 표현 방식을 혁신한 논문.

## 폴더 구조
```
Word2Vec/
├── README.md
├── paper_review.md
├── model.py        # CBOW, Skip-gram
├── train.py
├── experiments/
└── results/
```

## 두 가지 모델

### CBOW (Continuous Bag of Words)
주변 단어로 중심 단어를 예측
```
[the, quick, ___, fox, jumps] → "brown"
```

### Skip-gram
중심 단어로 주변 단어를 예측
```
"brown" → [the, quick, fox, jumps]
```

## 실행 방법
```bash
cd Paper_Implementation/Word2Vec
python train.py
```

## 핵심 contribution
1. **고품질 단어 임베딩** - 단어 간 의미적 유사도 학습
2. **벡터 산술 가능** - "King - Man + Woman ≈ Queen"
3. **빠른 학습** - 대용량 코퍼스에서도 학습 가능

## 단어 벡터의 매력
유사한 의미의 단어는 벡터 공간에서 가깝게 위치:
- "강아지" ≈ "개"
- "좋다" ≈ "훌륭하다"

## 참고
- [원본 논문 (arxiv)](https://arxiv.org/abs/1301.3781)
- [후속 논문 (Negative Sampling)](https://arxiv.org/abs/1310.4546)
