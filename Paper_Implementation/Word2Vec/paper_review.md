# Word2Vec 논문 리뷰

## 기본 정보
- **제목**: Efficient Estimation of Word Representations in Vector Space
- **저자**: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (Google)
- **발표**: ICLR 2013
- **링크**: https://arxiv.org/abs/1301.3781

## 한 줄 요약
단어를 의미를 담은 고정 차원의 벡터로 변환하는 효율적인 모델로, 현대 NLP의 임베딩 기반 모델들의 시초.

## 배경

### 기존 단어 표현의 한계
- **One-hot Encoding**: 차원이 너무 큼 (어휘 크기), 단어 간 관계 표현 불가
- **단어 빈도 기반 (TF-IDF)**: 의미적 유사성 표현 불가

### 필요한 것
- 차원이 작고 (예: 100~300)
- 단어 간 유사성을 잘 표현하는 벡터

## 핵심 아이디어

### 분포 가설 (Distributional Hypothesis)
> "비슷한 문맥에 등장하는 단어는 비슷한 의미를 가진다."

이 가설을 신경망으로 학습.

### CBOW (Continuous Bag of Words)
주변 단어들로 중심 단어를 예측하는 모델.

```
입력: 주변 단어들의 임베딩 평균
출력: 중심 단어
```

### Skip-gram
중심 단어로 주변 단어들을 예측하는 모델.

```
입력: 중심 단어 임베딩
출력: 주변 단어들
```

CBOW vs Skip-gram:
- **CBOW**: 빠름, 자주 나오는 단어에 강함
- **Skip-gram**: 더 정확, 희귀 단어에 강함

## 모델 구조

### Skip-gram 구조
```
중심 단어 (one-hot) 
   ↓ Embedding 행렬 W (V × D)
임베딩 벡터 (D 차원)
   ↓ 출력 행렬 W' (D × V)
softmax → 주변 단어 확률
```

V: 어휘 크기, D: 임베딩 차원 (보통 100~300)

## 학습 효율화

### 1. Hierarchical Softmax
- 일반 softmax는 모든 단어에 대해 계산 (O(V))
- Hierarchical softmax는 트리 구조 사용 (O(log V))

### 2. Negative Sampling (후속 논문)
- 정답 단어 + 무작위 negative 단어 몇 개만 학습
- 매우 빠르고 성능 좋음

## 신기한 결과

### 단어 유사도
- vector("강아지") ≈ vector("개")
- vector("happy") ≈ vector("joyful")

### 벡터 산술
$$\vec{King} - \vec{Man} + \vec{Woman} \approx \vec{Queen}$$
$$\vec{Paris} - \vec{France} + \vec{Korea} \approx \vec{Seoul}$$

→ 단어 벡터가 의미적/문법적 관계를 표현함을 보임

## 한계점

### 1. 다의어 처리 불가
"bank"가 은행/강둑 모두에서 같은 벡터 사용.<br>
→ 이후 ELMo, BERT 같은 contextual embedding으로 해결.

### 2. OOV 문제
학습되지 않은 단어는 표현 불가.<br>
→ FastText는 subword를 사용해 해결.

### 3. 정적 임베딩
한 단어 = 한 벡터. 문맥에 따라 변하지 않음.

## 느낀 점

### 1. 단순함의 위력
복잡한 RNN/LSTM 없이 단순한 신경망으로 큰 성과.<br>
"Distributional Hypothesis"라는 직관을 잘 구현.

### 2. NLP의 전환점
이전: 수작업 특성 (POS tagging, parsing)<br>
이후: 임베딩 → DL 모델

### 3. 표현 학습의 시작
"의미를 벡터에 담는다"는 패러다임이 시작됨.<br>
이는 BERT, GPT 같은 현대 언어 모델로 이어짐.

## 면접 질문

**Q1: Word2Vec의 핵심 contribution은?**<br>
A: 단어를 의미를 담은 고차원 벡터로 변환하는 효율적인 방법을 제안.<br>
이를 통해 NLP에서 임베딩 기반 딥러닝이 가능해졌다.

**Q2: CBOW와 Skip-gram의 차이는?**<br>
A: CBOW는 주변 단어 → 중심 단어, Skip-gram은 중심 단어 → 주변 단어.<br>
CBOW가 빠르지만 Skip-gram이 희귀 단어에 더 정확하다.

**Q3: 분포 가설이란?**<br>
A: "비슷한 문맥에 나타나는 단어는 비슷한 의미"라는 가설.<br>
Word2Vec은 이를 신경망으로 학습한다.

**Q4: Word2Vec의 한계와 후속 연구는?**<br>
A: 다의어 처리 불가 → ELMo (양방향 LSTM), BERT (Transformer) 등이 문맥 기반 임베딩 제공.<br>
OOV 문제 → FastText의 subword.
