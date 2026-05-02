# 원본 결과 재현 실험

## 목표
U-Net을 PyTorch로 구현하여 세그멘테이션 데이터셋에서 성능을 확인한다.

## 데이터셋 후보
- **Carvana** (캐글) - 자동차 배경 제거, 무료
- **Oxford Pets** - 동물 세그멘테이션
- **ISIC 2018** - 피부 병변 (의료)
- **Cityscapes** - 도시 환경 (자율주행)

## 실험 설정 (Carvana 기준)

### 모델
- U-Net (in_channels=3, out_channels=1)
- features=[64, 128, 256, 512]

### 하이퍼파라미터
| 항목 | 값 |
| --- | --- |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 16 |
| Epochs | 50 |
| Loss | BCE + Dice Loss |

### 평가 지표
- **Dice Score**: 세그멘테이션 정확도 (0~1, 높을수록 좋음)
- **IoU (Intersection over Union)**

$$Dice = \frac{2|A \cap B|}{|A| + |B|}$$
$$IoU = \frac{|A \cap B|}{|A \cup B|}$$

## 실험 결과

### 학습 곡선
| Epoch | Train Loss | Val Dice |
| --- | --- | --- |
| 10 |  |  |
| 30 |  |  |
| 50 |  |  |

### 시각화
- 원본 이미지
- 정답 마스크
- 예측 마스크
- 비교 그림

## 분석


## 결론
