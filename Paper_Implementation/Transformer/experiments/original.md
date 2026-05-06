# 원본 결과 재현 실험

## 목표
Transformer를 PyTorch로 구현하여 시퀀스 처리 능력을 확인한다.<br>
번역 데이터셋(WMT) 대신 **시퀀스 뒤집기 task**로 데모 실행.

## 실험 설정

### 데이터셋 (합성)
- **Sequence Reverse Task**
- 입력: 길이 8의 무작위 토큰 시퀀스
- 출력: 입력의 역순 ([BOS] + 역순 + [EOS])
- 학습: 2000개 샘플

### 모델 (Small Transformer)
| 항목 | 값 |
| --- | --- |
| Vocab Size | 20 |
| d_model | 64 |
| Heads | 4 |
| Layers (Enc/Dec) | 2 |
| d_ff | 128 |

### 하이퍼파라미터
| 항목 | 값 |
| --- | --- |
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Batch Size | 32 |
| Epochs | 20 |

## 실험 결과

### 학습 결과 (주요 epoch)
| Epoch | Train Loss | Train Acc |
| --- | --- | --- |
| 2 | 1.8939 | 0.3941 |
| 5 | 0.3693 | 0.8943 |
| 10 | 0.1418 | 0.9574 |
| 15 | 0.1035 | 0.9674 |
| 20 | 0.0824 | 0.9757 |

### 최종 성능
- **Train Accuracy: 97.57%**
- 시퀀스 뒤집기를 매우 잘 학습

## 분석

### 빠른 학습 속도
- 5 epoch만에 89% 달성
- 20 epoch에 97.5% 도달
- Self-Attention이 시퀀스 위치 정보를 효과적으로 처리

### Transformer의 강점 확인
- Sequence Reverse는 RNN으로는 어려운 task (긴 의존성 필요)
- Self-Attention이 모든 위치를 직접 참조하므로 자연스럽게 해결
- Cross-Attention이 인코더 출력을 디코더에 잘 전달

### 합성 task의 한계
- 단순 패턴 학습이라 실제 번역의 복잡성은 표현 못함
- 어휘 크기, 문법, 의미 모호성 등 미적용

## 다음 단계
1. **Multi30k** (DE-EN) 번역 데이터셋으로 실제 학습
2. BLEU score로 번역 품질 평가
3. Greedy decoding vs Beam search 비교
4. 어텐션 시각화

## 결론
Transformer 구현이 정상 동작하며 시퀀스 변환 task를 효과적으로 학습.<br>
Self-Attention의 위력을 직접 확인.
