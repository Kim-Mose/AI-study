# BERT PyTorch 구현

> **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (Devlin et al., 2018)

## 개요
Transformer 인코더만 사용해 양방향 문맥을 학습하는 사전학습 언어 모델.<br>
NLP의 pre-training + fine-tuning 패러다임을 확립했다.

## 폴더 구조
```
BERT/
├── README.md
├── paper_review.md
├── model.py        # BERT, MLM, NSP
├── train.py
├── experiments/
└── results/
```

## 모델 구조
```
입력: [CLS] 문장1 [SEP] 문장2 [SEP]
   ↓ Token + Segment + Position Embedding
   ↓
Transformer Encoder × N (BERT-base: 12, BERT-large: 24)
   ↓
출력: 각 토큰의 표현
```

## 사전학습 Task
1. **MLM (Masked Language Model)** - 15% 토큰 가리고 예측
2. **NSP (Next Sentence Prediction)** - 두 문장이 연속인지 분류

## 모델 크기
| 모델 | Layers | Hidden | Heads | Params |
| --- | --- | --- | --- | --- |
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

## 실행 방법
```bash
cd Paper_Implementation/BERT
python model.py
```

실제 사전학습은 자원이 매우 많이 필요하므로,<br>
보통 huggingface transformers의 사전학습 모델을 가져와 fine-tuning한다.

## 핵심 contribution
1. **양방향 문맥** - 좌우 문맥 모두 사용 (GPT는 단방향)
2. **Pre-training + Fine-tuning** 패러다임
3. **광범위한 NLP task에서 SOTA**
4. **사전학습된 모델 공개** - NLP 민주화

## 참고
- [원본 논문 (arxiv)](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
