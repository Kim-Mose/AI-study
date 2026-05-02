# 원본 결과 재현 실험

## 목표
Transformer를 PyTorch로 구현하여 번역 task에서 동작을 확인한다.

## 데이터셋 후보
- **Multi30k** (DE-EN) - 작은 번역 데이터셋, 빠른 실험용
- **IWSLT** - 중간 규모
- **WMT** - 대규모 (논문 원본)

## 실험 설정 (Multi30k 기준)

### 모델 (Base 모델)
| 항목 | 값 |
| --- | --- |
| d_model | 512 |
| Heads | 8 |
| Layers (Enc/Dec) | 6 |
| d_ff | 2048 |
| Dropout | 0.1 |

### 하이퍼파라미터
| 항목 | 값 |
| --- | --- |
| Optimizer | Adam (β1=0.9, β2=0.98) |
| Warmup Steps | 4000 |
| Batch Size | 32 |
| Epochs | 30 |

### 평가 지표
- **BLEU Score** - 번역 품질 측정

## 실험 결과

### 학습 곡선
| Epoch | Train Loss | Val BLEU |
| --- | --- | --- |
| 5 |  |  |
| 15 |  |  |
| 30 |  |  |

### 번역 예시
| 입력 (DE) | 정답 (EN) | 모델 출력 (EN) |
| --- | --- | --- |
|  |  |  |

## 분석


## 결론
