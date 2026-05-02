# 원본 결과 재현 실험

## 목표
BERT를 PyTorch로 구현하고, 작은 규모로 사전학습 또는 fine-tuning을 시도한다.

## 현실적 접근
BERT 사전학습은 자원이 매우 많이 필요(수만 달러).<br>
보통 두 가지 접근:
1. **작은 BERT 직접 사전학습** (학습 동작 확인용)
2. **사전학습된 BERT를 fine-tuning** (실제 활용)

## 옵션 1: 작은 BERT 사전학습

### 데이터셋
- 위키피디아 일부 또는 BookCorpus 일부

### 모델 (mini BERT)
| 항목 | 값 |
| --- | --- |
| Layers | 4 |
| Hidden | 256 |
| Heads | 4 |

### 결과
| Step | MLM Loss | NSP Acc |
| --- | --- | --- |
|  |  |  |

## 옵션 2: 사전학습 BERT Fine-tuning

### 데이터셋
- **NSMC** (한국어 영화 리뷰 감성 분석)
- **SST-2** (영어 감성 분석)
- **KLUE** (한국어 NLU 벤치마크)

### 모델
- huggingface의 `bert-base-multilingual-cased` 또는 `klue/bert-base`

### 하이퍼파라미터
| 항목 | 값 |
| --- | --- |
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Epochs | 3 |

### 결과 (NSMC 감성 분석 기준)
| Epoch | Train Acc | Val Acc |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |

## 분석


## 결론
