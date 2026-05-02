# BERT 논문 리뷰

## 기본 정보
- **제목**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **저자**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google)
- **발표**: NAACL 2019 (arxiv 2018)
- **링크**: https://arxiv.org/abs/1810.04805

## 한 줄 요약
Transformer 인코더로 양방향 문맥을 학습한 사전학습 언어 모델로, NLP에 pre-training + fine-tuning 패러다임을 확립한 논문.

## 배경

### 기존 언어 모델의 한계
- **단방향**: GPT는 좌→우만 봄
- **얕은 양방향**: ELMo는 LSTM의 양방향 출력을 단순 결합
- **깊은 양방향이 필요**

### 왜 BERT가 필요했나?
문장 이해에는 양쪽 문맥이 모두 필요.
- "그 은행은 ___이 많다" → 빈칸 예측에 좌우 문맥 다 필요

## 핵심 아이디어

### 1. Bidirectional Self-Attention
Transformer 인코더는 모든 토큰이 모든 토큰을 볼 수 있음.<br>
GPT의 디코더는 미래 마스킹으로 단방향이지만, BERT는 인코더 사용.

### 2. Masked Language Model (MLM)
"양방향이면 자기 자신을 보는 문제 어떻게?" → **마스킹으로 해결**

학습 방식:
- 15% 토큰을 랜덤 선택
  - 80%: [MASK] 토큰으로 대체
  - 10%: 다른 단어로 대체
  - 10%: 그대로 둠
- 모델은 원본 단어 예측

### 3. Next Sentence Prediction (NSP)
두 문장 (A, B)가 주어지면:
- 50%: 실제 연속 문장
- 50%: 무작위 문장

[CLS] 토큰의 출력으로 이진 분류.<br>
(나중에 RoBERTa 등에서는 NSP 효과가 미미함이 밝혀짐)

### 4. 입력 표현
```
입력: [CLS] my dog is cute [SEP] he likes play ##ing [SEP]
임베딩 = Token + Segment + Position
```

- **[CLS]**: 분류용 토큰
- **[SEP]**: 문장 구분
- **WordPiece** 토크나이저 (subword)

## 모델 구조

### BERT-base
- Layers: 12
- Hidden Size: 768
- Heads: 12
- Parameters: 110M

### BERT-large
- Layers: 24
- Hidden Size: 1024
- Heads: 16
- Parameters: 340M

## 학습

### Pre-training
- BookCorpus (800M words) + Wikipedia (2,500M words)
- Batch Size: 256
- Steps: 1M
- TPU 4개로 약 4일

### Fine-tuning
다양한 task에 적용:
- 분류: [CLS] 출력으로 분류
- QA: 시작/끝 위치 예측
- NER: 각 토큰 출력으로 분류

## 실험 결과 (GLUE 벤치마크)

| 모델 | GLUE 평균 |
| --- | --- |
| Pre-OpenAI SOTA | 75.1 |
| GPT-1 | 75.1 |
| **BERT-base** | **79.6** |
| **BERT-large** | **82.1** |

→ 11개 task에서 SOTA

## 한계점

### 1. 사전학습 비용
BERT-large 학습에 수만 달러의 컴퓨팅 자원 필요.

### 2. NSP의 효과 미미
RoBERTa 등에서 NSP가 별 효과 없음을 보임.<br>
실제로는 MLM만으로 충분.

### 3. 마스크 토큰 불일치
사전학습에는 [MASK]가 있지만 fine-tuning에는 없음 → 약간의 불일치.

### 4. 생성 task에 약함
인코더만 있어 텍스트 생성에는 부적합.<br>
→ T5, BART 같은 인코더-디코더 모델로 발전.

## 느낀 점

### 1. Pre-training의 위력
큰 코퍼스로 사전학습한 모델을 작은 데이터로 fine-tuning하는 방식이 NLP 표준이 됨.<br>
이는 Computer Vision에서도 ViT 등으로 확장.

### 2. 양방향의 가치
단방향 GPT보다 양방향 BERT가 이해 task에서 강함.<br>
다만 생성 task는 GPT가 강함 → 용도에 따라 선택.

### 3. Hugging Face의 등장
BERT 이후 Hugging Face Transformers가 사전학습 모델을 쉽게 사용 가능하게 만듦.<br>
NLP 진입 장벽이 매우 낮아짐.

## 면접 질문

**Q1: BERT와 GPT의 차이는?**<br>
A: BERT는 Transformer 인코더 (양방향), GPT는 디코더 (단방향).<br>
BERT는 이해 task에 강하고, GPT는 생성 task에 강하다.

**Q2: MLM에서 왜 80/10/10 비율을 쓰나?**<br>
A: 항상 [MASK]만 쓰면 fine-tuning시 [MASK]가 없을 때 일치 안 됨.<br>
다양한 변형으로 모델이 더 robust해진다.

**Q3: NSP는 효과가 있나?**<br>
A: 원논문에선 도움된다 했지만, RoBERTa, ALBERT에서 NSP 제거해도 성능 유지/향상.<br>
실제로는 효과가 미미하다고 알려짐.

**Q4: BERT를 어떻게 fine-tuning하는가?**<br>
A: task별로 출력 layer를 추가하고 작은 학습률(2e-5 등)로 전체 모델을 학습.<br>
분류: [CLS]에 분류 헤드, QA: 시작/끝 위치 예측 헤드.

**Q5: BERT의 후속 발전은?**<br>
A: RoBERTa (NSP 제거, 더 많은 데이터), ALBERT (파라미터 공유),<br>
DistilBERT (작은 모델), DeBERTa (개선된 어텐션) 등.
