# Ablation 실험

## 실험 1: 패치 크기 비교

### 가설
패치가 작을수록 표현력 ↑이지만 계산량 ↑.

### 결과
| Patch Size | 패치 수 | Test Acc | 학습 시간 |
| --- | --- | --- | --- |
| 2 | 256 |  |  |
| 4 (원본) | 64 |  |  |
| 8 | 16 |  |  |

## 실험 2: 모델 크기

### 결과
| 모델 | Embed Dim | Depth | Test Acc |
| --- | --- | --- | --- |
| Tiny | 96 | 4 |  |
| Small (원본) | 192 | 6 |  |
| Base | 384 | 12 |  |

## 실험 3: ViT vs ResNet (작은 데이터셋)

### 가설
CIFAR-10 같은 작은 데이터셋에선 ResNet이 ViT보다 강할 것.

### 결과
| 모델 | 파라미터 | Test Acc |
| --- | --- | --- |
| ResNet18 |  |  |
| ViT-Small |  |  |

## 실험 4: 데이터 증강 효과

### 가설
ViT는 데이터에 민감하므로 강한 증강이 효과적일 것.

### 결과
| 증강 | Test Acc |
| --- | --- |
| 기본 (Crop, Flip) |  |
| + RandAugment |  |
| + MixUp/CutMix |  |

## 실험 5: Positional Embedding 비교

### 결과
| PE 방식 | Test Acc |
| --- | --- |
| 없음 |  |
| Learned 1D (원본) |  |
| Learned 2D |  |
| Sinusoidal |  |

## 실험 6: [CLS] 토큰 vs Global Average Pooling

### 가설
[CLS] 토큰 대신 모든 패치의 평균을 사용해도 비슷한 성능?

### 결과
| 풀링 방식 | Test Acc |
| --- | --- |
| [CLS] 토큰 (원본) |  |
| Global Average Pooling |  |

## 종합 결론
- ViT는 작은 데이터셋에서 CNN보다 약하다는 점 확인
- 강한 증강과 정규화가 필요
- 큰 사전학습 모델 사용이 실무적 정답
