# 원본 결과 재현 실험

## 목표
Word2Vec(CBOW)을 PyTorch로 구현하여 단어 임베딩 학습을 확인한다.<br>
실제 대규모 코퍼스 대신 **간단한 예시 문장**으로 데모 실행.

## 실험 설정

### 데이터셋 (예시 코퍼스)
```
"the quick brown fox jumps over the lazy dog the cat sits on the mat the dog barks at the cat"
```
- 총 14개 unique 단어 (vocab size)

### 모델
- **CBOW** (Continuous Bag of Words)
- 주변 단어 → 중심 단어 예측

### 하이퍼파라미터
| 항목 | 값 |
| --- | --- |
| Embedding Dim | 50 |
| Window Size | 2 |
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Epochs | 100 |

## 실험 결과

### 학습 결과 (주요 epoch)
| Epoch | Loss |
| --- | --- |
| 10 | 0.2626 |
| 30 | 0.0242 |
| 50 | 0.0086 |
| 70 | 0.0043 |
| 100 | 0.0020 |

### 최종 성능
- **최종 Loss: 0.0020** (잘 수렴)
- 임베딩 shape: (14, 50) - 14개 단어, 50차원

### 학습 곡선
`results/training_curve.png` 참고

## 분석

### 작은 데이터에서의 한계
- 14개 단어로는 의미적 관계 학습이 부족
- 실제 Word2Vec의 진가(예: King - Man + Woman ≈ Queen)는 대규모 코퍼스에서 발현
- 본 실험은 모델 동작 확인 수준

### 정상 동작 확인
- Loss가 안정적으로 감소
- CBOW 구조가 정상 작동
- 임베딩 행렬이 적절한 shape으로 생성됨

## 다음 단계
1. **Wikipedia 한국어 dump** 또는 **NSMC** 데이터로 실제 학습
2. CBOW vs Skip-gram 비교
3. 단어 유사도 계산 (코사인 유사도)
4. 벡터 산술 검증
5. t-SNE로 임베딩 시각화

## 결론
Word2Vec(CBOW) 구조가 정상 동작함을 확인.<br>
의미 있는 임베딩을 얻으려면 대규모 코퍼스 + 더 많은 학습 필요.
